"""Regenerate every figure used in the thesis from a single analysis run.

All subject labels are anonymised codes, and behavioural states are re-indexed by
mean activity within each subject so that state numbers are comparable across the
per-subject models.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from src import models_hsmm

RUN = os.environ.get("THESIS_RUN_VERSION", "v6")
PROCESSED = Path("data/processed") / RUN
OUTPUTS = Path("past-runs") / RUN / "outputs"
FIGDIR = Path(os.environ.get("THESIS_FIGURE_DIR", OUTPUTS / "figures"))
FIGDIR.mkdir(parents=True, exist_ok=True)

REPRESENTATIVE = "S15"
STATE_LABELS = ["Deep rest", "Light rest", "Moderate activity", "High activity"]
STATE_COLORS = ["#2c7fb8", "#7fcdbb", "#fec44f", "#d95f0e"]
DWELL_COL = "hmm_state_filtered"
BIN_HOURS = 0.25

# The thesis typesets figures at \textwidth = 390pt = 5.42in. Keeping figures close
# to that width means they are barely downscaled on the page, so the point sizes
# below are very nearly what the reader sees in print.
FIG_W = 6.0

plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.6,
})


def load() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "hmm_results_improved.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "subject_code" not in df.columns:
        raise SystemExit("subject_code missing; re-run the preprocessing stage")
    return df


def rank_states(df: pd.DataFrame, col: str = DWELL_COL) -> pd.DataFrame:
    """Re-index states by mean activity within each subject."""
    out = []
    for _, g in df.groupby("subject_code", sort=False):
        order = g.groupby(col)["activity"].mean().sort_values()
        remap = {old: new for new, old in enumerate(order.index)}
        g = g.copy()
        g["state"] = g[col].map(remap)
        out.append(g)
    return pd.concat(out)


def bouts(states: np.ndarray) -> pd.DataFrame:
    s = np.asarray(states)
    s = s[~pd.isna(s)].astype(int)
    if s.size == 0:
        return pd.DataFrame(columns=["state", "hours"])
    edges = np.concatenate(([0], np.where(np.diff(s) != 0)[0] + 1, [s.size]))
    return pd.DataFrame({
        "state": s[edges[:-1]],
        "hours": np.diff(edges) * BIN_HOURS,
    })


def save(fig, name: str) -> None:
    path = FIGDIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"wrote {path}")


# --------------------------------------------------------------------------- #

def fig_posture_validation(df: pd.DataFrame) -> None:
    d = rank_states(df).dropna(subset=["state"])
    g = d.groupby("state").agg(
        activity=("activity", "mean"),
        mag=("activity_xyz", "mean"),
        sd=("activity_xyz", "std"),
    )

    fig, axes = plt.subplots(3, 1, figsize=(FIG_W, 7.4), sharex=True)

    axes[0].bar(g.index, g["activity"], color=STATE_COLORS)
    axes[0].set_ylabel("Mean active minutes\nper epoch")
    axes[0].set_title("(a) Activity by state")

    axes[1].bar(g.index, g["mag"], color=STATE_COLORS)
    axes[1].set_ylabel("Mean posture\nmagnitude (ADC)")
    axes[1].set_ylim(bottom=150)
    axes[1].set_title("(b) Magnitude falls as activity rises")

    axes[2].bar(g.index, g["sd"], color=STATE_COLORS)
    axes[2].set_ylabel("SD of posture\nmagnitude (ADC)")
    axes[2].set_ylim(bottom=90)
    axes[2].set_title("(c) Variability rises as activity rises")

    axes[2].set_xticks(range(4))
    axes[2].set_xticklabels(STATE_LABELS)

    fig.suptitle("Posture channel validates the state ordering (not used in fitting)")
    fig.tight_layout()
    save(fig, "posture_state_validation.png")


def fig_model_selection(df: pd.DataFrame) -> None:
    """Refit the representative subject across K to show the selection criteria."""
    from hmmlearn import hmm

    sub = df[df["subject_code"] == REPRESENTATIVE]
    x = sub["activity_log_standardized"].dropna().to_numpy().reshape(-1, 1)

    ks, aics, bics, seps = [], [], [], []
    for k in (2, 3, 4, 5):
        m = hmm.GaussianHMM(n_components=k, covariance_type="diag",
                            n_iter=100, random_state=42)
        m.fit(x)
        ll = m.score(x)
        # transitions + means + variances + K-1 free initial-state probabilities
        p = k * k + 2 * k - 1
        ks.append(k)
        aics.append(-2 * ll + 2 * p)
        bics.append(-2 * ll + p * np.log(len(x)))
        mu = m.means_.ravel()
        seps.append(float(np.std(mu) / np.abs(np.mean(mu))) if np.mean(mu) else np.nan)

    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 6.4))
    axes[0].plot(ks, aics, "o-", label="AIC")
    axes[0].plot(ks, bics, "s-", label="BIC")
    axes[0].axvline(4, color="crimson", ls="--", alpha=0.7, label="selected $K=4$")
    axes[0].set_xlabel("Number of states $K$")
    axes[0].set_ylabel("Information criterion")
    axes[0].set_xticks(ks)
    axes[0].legend()
    axes[0].set_title(f"(a) Model selection, subject {REPRESENTATIVE}")

    axes[1].bar([str(k) for k in ks], seps, color=STATE_COLORS[: len(ks)])
    axes[1].set_xlabel("Number of states $K$")
    axes[1].set_ylabel("State separation score")
    axes[1].set_title("(b) State separation")

    fig.tight_layout()
    save(fig, "model_selection.png")


def fig_transition_matrix(df: pd.DataFrame) -> None:
    d = rank_states(df)
    sub = d[d["subject_code"] == REPRESENTATIVE]["state"].dropna().astype(int).to_numpy()

    k = 4
    T = np.zeros((k, k))
    for a, b in zip(sub[:-1], sub[1:]):
        T[a, b] += 1
    T = T / T.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 8.2),
                             gridspec_kw={"height_ratios": [1.9, 1]})

    im = axes[0].imshow(T, cmap="YlOrRd", vmin=0, vmax=1)
    for i in range(k):
        for j in range(k):
            axes[0].text(j, i, f"{T[i, j]:.3f}", ha="center", va="center",
                         color="white" if T[i, j] > 0.5 else "black", fontsize=9)
    axes[0].set_xticks(range(k))
    axes[0].set_yticks(range(k))
    axes[0].set_xticklabels(STATE_LABELS, rotation=20, ha="right")
    axes[0].set_yticklabels(STATE_LABELS)
    axes[0].set_xlabel("To state")
    axes[0].set_ylabel("From state")
    axes[0].set_title(f"(a) Transition probabilities, subject {REPRESENTATIVE}")
    axes[0].grid(False)
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    diag = np.diag(T)
    axes[1].bar(range(k), diag, color=STATE_COLORS)
    axes[1].axhline(diag.mean(), color="crimson", ls="--",
                    label=f"mean = {diag.mean():.3f}")
    axes[1].set_xticks(range(k))
    axes[1].set_xticklabels(STATE_LABELS, rotation=20, ha="right")
    axes[1].set_ylabel("Self-transition\nprobability")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].set_title("(b) State persistence")

    fig.tight_layout()
    save(fig, "transition_matrix.png")


def fig_dwell_times(df: pd.DataFrame) -> None:
    d = rank_states(df)
    allb = []
    for _, g in d.groupby("subject_code", sort=False):
        allb.append(bouts(g["state"].to_numpy()))
    b = pd.concat(allb)

    fig, axes = plt.subplots(4, 1, figsize=(FIG_W, 8.2), sharex=True)
    for k, ax in enumerate(axes):
        v = b[b["state"] == k]["hours"]
        ax.hist(v, bins=np.arange(0, 12.5, 0.25), color=STATE_COLORS[k],
                edgecolor="white", linewidth=0.3)
        ax.axvline(v.mean(), color="crimson", ls="--", label=f"mean {v.mean():.2f} h")
        ax.axvline(v.median(), color="navy", ls=":", label=f"median {v.median():.2f} h")
        ax.set_title(f"({chr(97 + k)}) {STATE_LABELS[k]}  ($n$ = {len(v):,} bouts)")
        ax.set_ylabel("Frequency")
        ax.set_xlim(0, 12)
        ax.legend()
    axes[-1].set_xlabel("Bout duration (h)")

    fig.suptitle("Bout duration distributions, all subjects pooled")
    fig.tight_layout()
    save(fig, "dwell_times.png")


def fig_bout_normality(df: pd.DataFrame) -> None:
    d = rank_states(df)
    allb = pd.concat(bouts(g["state"].to_numpy())
                     for _, g in d.groupby("subject_code", sort=False))

    fig, axes = plt.subplots(4, 2, figsize=(FIG_W, 7.4))
    for k in range(4):
        v = allb[allb["state"] == k]["hours"].to_numpy()

        ax = axes[k, 0]
        ax.hist(v, bins=60, density=True, color=STATE_COLORS[k], alpha=0.8)
        xs = np.linspace(v.min(), v.max(), 200)
        ax.plot(xs, stats.norm.pdf(xs, v.mean(), v.std()), "r-", lw=1.2)
        ax.set_title(f"({chr(97 + k)}) {STATE_LABELS[k]}: "
                     f"$n$={len(v):,}, skew={stats.skew(v):.2f}")
        ax.set_xlabel("Duration (h)")
        ax.set_ylabel("Density")

        ax = axes[k, 1]
        stats.probplot(v, dist="norm", plot=ax)
        ax.set_title(f"({chr(97 + k)}) Q--Q plot")
        ax.get_lines()[0].set_markersize(1.5)
        ax.get_lines()[0].set_color(STATE_COLORS[k])
        ax.get_lines()[1].set_color("crimson")
        ax.set_xlabel("Theoretical quantiles")
        ax.set_ylabel("Observed")

    fig.suptitle("Bout durations depart strongly from normality in every state")
    fig.tight_layout()
    save(fig, "bout_duration_normality.png")


def fig_state_probability_by_hour(df: pd.DataFrame) -> None:
    d = rank_states(df).dropna(subset=["state"])
    d["hour"] = d["timestamp"].dt.hour

    props = (d.groupby(["hour", "state"]).size().unstack(fill_value=0))
    props = props.div(props.sum(axis=1), axis=0)

    hourly = d.groupby("hour")["activity"].agg(["mean", "sem"])

    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 6.6), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})

    axes[0].stackplot(props.index, [props[c] for c in props.columns],
                      labels=STATE_LABELS, colors=STATE_COLORS, alpha=0.9)
    axes[0].set_ylabel("State probability")
    axes[0].set_ylim(0, 1)
    axes[0].legend(loc="upper right", ncol=2, fontsize=8)
    axes[0].set_title("Behavioural state occupancy by hour of day")

    axes[1].plot(hourly.index, hourly["mean"], "o-", color="#08519c")
    axes[1].fill_between(hourly.index, hourly["mean"] - hourly["sem"],
                         hourly["mean"] + hourly["sem"], alpha=0.3, color="#08519c")
    axes[1].set_xlabel("Hour of day")
    axes[1].set_ylabel("Mean active\nminutes per epoch")
    axes[1].set_xticks(range(0, 24, 2))
    axes[1].set_xlim(0, 23)

    for ax in axes:
        ax.axvspan(0, 6, color="#3b4a6b", alpha=0.10)
        ax.axvspan(18, 23, color="#3b4a6b", alpha=0.10)

    fig.tight_layout()
    save(fig, "state_probability_by_hour.png")


def fig_between_subject(df: pd.DataFrame) -> None:
    d = rank_states(df).dropna(subset=["state"])
    codes = sorted(d["subject_code"].unique())

    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 7.2))

    # ActMindata is a bounded count with a median of zero, so a boxplot degenerates;
    # subject means with 95% CIs are more informative.
    stats_by_subject = d.groupby("subject_code")["activity"].agg(["mean", "sem"]).loc[codes]
    axes[0].bar(codes, stats_by_subject["mean"],
                yerr=1.96 * stats_by_subject["sem"],
                color="#9ecae1", edgecolor="#3182bd", capsize=2)
    axes[0].axhline(stats_by_subject["mean"].mean(), color="crimson", ls="--",
                    label=f"cohort mean = {stats_by_subject['mean'].mean():.2f}")
    axes[0].set_ylabel("Mean active minutes\nper epoch")
    axes[0].set_xlabel("Subject")
    axes[0].tick_params(axis="x", rotation=60)
    axes[0].legend()
    axes[0].set_title("(a) Mean activity by subject (95% CI)")

    props = (d.groupby(["subject_code", "state"]).size().unstack(fill_value=0))
    props = props.div(props.sum(axis=1), axis=0).loc[codes]
    bottom = np.zeros(len(codes))
    for k in range(4):
        vals = props[k].to_numpy() if k in props.columns else np.zeros(len(codes))
        axes[1].bar(codes, vals, bottom=bottom, color=STATE_COLORS[k],
                    label=STATE_LABELS[k])
        bottom += vals
    axes[1].set_ylabel("Proportion of time")
    axes[1].set_xlabel("Subject")
    axes[1].tick_params(axis="x", rotation=60)
    axes[1].set_ylim(0, 1)
    axes[1].legend(ncol=2, fontsize=8)
    axes[1].set_title("(b) State allocation by subject")

    fig.tight_layout()
    save(fig, "between_subject_comparison_v2.png")


def _sex_map(df: pd.DataFrame) -> pd.Series:
    meta_path = Path("data/raw/ACT_Wet_Dry_Seasons_BR.hourly.csv")
    sex = pd.Series(dtype=object)
    if meta_path.exists():
        meta = pd.read_csv(meta_path, usecols=["subject", "sex"])
        sex = meta.groupby("subject")["sex"].first()
    ov_path = Path("data/raw/subject_sex_overrides.csv")
    if ov_path.exists():
        ov = pd.read_csv(ov_path, usecols=["subject", "sex"])
        ov = ov.drop_duplicates("subject").set_index("subject")["sex"]
        sex = sex.combine_first(ov) if len(sex) else ov
    return sex


def fig_sex_effects(df: pd.DataFrame) -> None:
    d = rank_states(df).dropna(subset=["state"])
    d["sex"] = d["subject"].map(_sex_map(df))
    d = d.dropna(subset=["sex"])
    if d.empty:
        logger.warning("no sex metadata; skipping sex figure")
        return

    fig = plt.figure(figsize=(FIG_W, 7.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, :])]

    subj = d.groupby(["subject_code", "sex"])["activity_log"].mean().reset_index()
    groups = [subj[subj["sex"] == s]["activity_log"].to_numpy() for s in ("Female", "Male")]
    bp = axes[0].boxplot(groups, labels=["Female", "Male"], patch_artist=True)
    for patch, colour in zip(bp["boxes"], ["#c994c7", "#9ecae1"]):
        patch.set_facecolor(colour)
    for i, g in enumerate(groups, start=1):
        axes[0].scatter(np.full(len(g), i) + np.random.uniform(-.06, .06, len(g)),
                        g, color="black", s=14, zorder=3)
    u, p = stats.mannwhitneyu(groups[0], groups[1])
    axes[0].set_ylabel("Subject mean\nlog activity")
    axes[0].set_title(f"(a) Activity by sex\nMann-Whitney $U$={u:.0f}, $p$={p:.3f}")

    props = d.groupby(["sex", "state"]).size().unstack(fill_value=0)
    props = props.div(props.sum(axis=1), axis=0)
    bottom = np.zeros(len(props))
    for k in range(4):
        vals = props[k].to_numpy() if k in props.columns else np.zeros(len(props))
        axes[1].bar(props.index, vals, bottom=bottom, color=STATE_COLORS[k],
                    label=STATE_LABELS[k])
        bottom += vals
    axes[1].set_ylabel("Proportion of time")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=7, loc="lower center")
    axes[1].set_title("(b) State allocation by sex")

    d["hour"] = d["timestamp"].dt.hour
    for s, colour in (("Female", "#c994c7"), ("Male", "#3182bd")):
        prof = d[d["sex"] == s].groupby("hour")["activity"].mean()
        axes[2].plot(prof.index, prof.to_numpy(), "o-", color=colour, label=s, ms=4)
    axes[2].set_xlabel("Hour of day")
    axes[2].set_ylabel("Mean active minutes")
    axes[2].set_xticks(range(0, 24, 2))
    axes[2].legend()
    axes[2].set_title("(c) Diel profile by sex")

    fig.tight_layout()
    save(fig, "sex_effects_v2.png")


def fig_cosinor(df: pd.DataFrame) -> None:
    from src import cosinor as cos

    d = df.dropna(subset=["activity_log"]).copy()
    d["hour"] = d["timestamp"].dt.hour
    hourly = d.groupby("hour")["activity_log"].mean()
    t = hourly.index.to_numpy(float)
    y = hourly.to_numpy()

    multi = cos.fit_multi_component_cosinor(t, y, n_components=5, base_period=24.0)
    single = cos.fit_cosinor(t, y, period=24.0)

    tt = np.linspace(0, 24, 400)
    mesor = multi["mesor"]

    def multi_curve(ts):
        out = np.full_like(ts, mesor)
        for c in multi["components"]:
            out = out + c["amplitude"] * np.cos(2 * np.pi * ts / c["period"] - c["acrophase_rad"])
        return out

    single_curve = single["mesor"] + single["amplitude"] * np.cos(
        2 * np.pi * tt / 24 - single["acrophase"])

    # --- cosinor_rhythm_v2: fit + harmonic decomposition ---
    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 6.8))
    axes[0].plot(t, y, "ko", ms=5, label="Hourly means")
    axes[0].plot(tt, multi_curve(tt), "-", color="crimson", lw=2,
                 label=f"5-component ($R^2$={multi['r_squared']:.3f})")
    axes[0].axhline(mesor, color="gray", ls=":", label=f"MESOR = {mesor:.3f}")
    axes[0].set_xlabel("Hour of day")
    axes[0].set_ylabel("Mean activity (log)")
    axes[0].set_xticks(range(0, 25, 4))
    axes[0].legend(fontsize=8)
    axes[0].set_title("(a) Multi-component cosinor fit")

    for c in multi["components"]:
        axes[1].plot(tt, c["amplitude"] * np.cos(2 * np.pi * tt / c["period"] - c["acrophase_rad"]),
                     label=f"{c['period']:.1f} h (A={c['amplitude']:.3f})")
    axes[1].axhline(0, color="black", lw=0.6)
    axes[1].set_xlabel("Hour of day")
    axes[1].set_ylabel("Amplitude contribution")
    axes[1].set_xticks(range(0, 25, 4))
    axes[1].legend(fontsize=8, ncol=2)
    axes[1].set_title("(b) Harmonic decomposition")
    fig.tight_layout()
    save(fig, "cosinor_rhythm_v2.png")

    # --- cosinor_multi_component: single vs multi, and per-subject R2 ---
    per_subject = []
    for code, g in d.groupby("subject_code", sort=False):
        h = g.groupby("hour")["activity_log"].mean()
        if len(h) < 12:
            continue
        tt_s = h.index.to_numpy(float)
        yy = h.to_numpy()
        m = cos.fit_multi_component_cosinor(tt_s, yy, n_components=5, base_period=24.0)
        s = cos.fit_cosinor(tt_s, yy, period=24.0)
        per_subject.append({"code": code, "multi": m["r_squared"], "single": s["r_squared"]})
    ps = pd.DataFrame(per_subject).sort_values("code")

    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 6.8))
    axes[0].plot(t, y, "ko", ms=5, label="Observed")
    axes[0].plot(tt, single_curve, "--", color="#3182bd", lw=1.8,
                 label=f"1-component ($R^2$={single['r_squared']:.3f})")
    axes[0].plot(tt, multi_curve(tt), "-", color="crimson", lw=2,
                 label=f"5-component ($R^2$={multi['r_squared']:.3f})")
    axes[0].set_xlabel("Hour of day")
    axes[0].set_ylabel("Mean activity (log)")
    axes[0].set_xticks(range(0, 25, 4))
    axes[0].legend(fontsize=8)
    axes[0].set_title("(a) Population-level model comparison")

    idx = np.arange(len(ps))
    axes[1].bar(idx - 0.2, ps["single"], 0.4, label="1-component", color="#3182bd")
    axes[1].bar(idx + 0.2, ps["multi"], 0.4, label="5-component", color="crimson")
    axes[1].set_xticks(idx)
    axes[1].set_xticklabels(ps["code"], rotation=60)
    axes[1].set_xlabel("Subject")
    axes[1].set_ylabel("$R^2$")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=8, ncol=2, loc="lower right")
    axes[1].set_title("(b) Per-subject fit quality")
    fig.tight_layout()
    save(fig, "cosinor_multi_component.png")


def fig_hsmm(df: pd.DataFrame) -> None:
    d = rank_states(df)
    codes = sorted(d["subject_code"].unique())[:5]

    rows, agreements, hmm_bouts, hsmm_bouts = [], [], [], []
    fitted = {}

    for code in codes:
        g = d[d["subject_code"] == code]
        x = g["activity_log_standardized"].to_numpy()
        hmm_all = g["state"].to_numpy(dtype=float)

        # Compare like with like: keep only steps where both models produce a state.
        mask = ~np.isnan(x) & ~np.isnan(hmm_all)
        res = models_hsmm.fit_gaussian_hsmm(x[mask], n_states=4, n_iter=30, max_duration=96)
        fitted[code] = res

        hmm_states = hmm_all[mask].astype(int)
        agreements.append(models_hsmm.state_sequence_agreement(hmm_states, res["states"]))
        hmm_bouts.append(models_hsmm.mean_bout_duration(hmm_states))
        hsmm_bouts.append(models_hsmm.mean_bout_duration(res["states"]))

        for k in range(4):
            rows.append({"code": code, "state": k, "lam": res["lambdas"][k],
                         "mean": res["means"][k]})

    lam = pd.DataFrame(rows)
    summary = lam.groupby("state")["lam"].mean()
    summary.to_frame("mean_lambda").assign(
        expected_hours=lambda t: t["mean_lambda"] * BIN_HOURS
    ).to_csv(OUTPUTS / "hsmm_duration_parameters.csv")

    # --- hsmm_duration_analysis ---
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, 6.4))

    axes[0, 0].boxplot([lam[lam["state"] == k]["lam"] * BIN_HOURS for k in range(4)],
                       labels=STATE_LABELS, patch_artist=True)
    axes[0, 0].tick_params(axis="x", rotation=30)
    axes[0, 0].set_ylabel("Expected duration (h)")
    axes[0, 0].set_title("(a) Fitted Poisson durations")

    axes[0, 1].scatter(lam["mean"], lam["lam"] * BIN_HOURS,
                       c=[STATE_COLORS[k] for k in lam["state"]], s=40)
    axes[0, 1].set_xlabel("State mean\n(standardised activity)")
    axes[0, 1].set_ylabel("Expected duration (h)")
    axes[0, 1].set_title("(b) Activity level vs duration")

    dd = np.arange(1, 25)
    for k in range(4):
        axes[1, 0].plot(dd * BIN_HOURS, stats.poisson.pmf(dd, summary[k]),
                        color=STATE_COLORS[k], label=f"{STATE_LABELS[k]} ($\\lambda$={summary[k]:.1f})")
    axes[1, 0].set_xlabel("Duration (h)")
    axes[1, 0].set_ylabel("Probability")
    axes[1, 0].legend(fontsize=7)
    axes[1, 0].set_title("(c) Fitted duration distributions")

    rep = fitted[codes[-1]]
    rep_b = bouts(rep["states"])
    for k in range(4):
        v = rep_b[rep_b["state"] == k]["hours"]
        if len(v):
            axes[1, 1].hist(v, bins=np.arange(0, 8.25, 0.25), alpha=0.6,
                            color=STATE_COLORS[k], label=STATE_LABELS[k])
    axes[1, 1].set_xlabel("Bout duration (h)")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].legend(fontsize=7)
    axes[1, 1].set_title(f"(d) HSMM bouts, subject {codes[-1]}")

    fig.tight_layout()
    save(fig, "hsmm_duration_analysis.png")

    # --- hsmm_vs_hmm_comparison ---
    fig = plt.figure(figsize=(FIG_W, 6.6))
    gs = fig.add_gridspec(2, 2)
    axes = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1])]

    axes[0].bar(codes, np.array(agreements) * 100, color="#3182bd")
    axes[0].axhline(np.mean(agreements) * 100, color="crimson", ls="--",
                    label=f"mean = {np.mean(agreements)*100:.1f}%")
    axes[0].set_ylabel("State sequence\nagreement (%)")
    axes[0].set_xlabel("Subject")
    axes[0].legend()
    axes[0].set_title("(a) HMM vs HSMM agreement")

    axes[1].scatter(hmm_bouts, hsmm_bouts, s=50, color="#6a51a3")
    lim = max(max(hmm_bouts), max(hsmm_bouts)) * 1.15
    axes[1].plot([0, lim], [0, lim], "k--", lw=1, label="$y = x$")
    axes[1].set_xlabel("HMM mean bout (h)")
    axes[1].set_ylabel("HSMM mean bout (h)")
    axes[1].set_xlim(0, lim)
    axes[1].set_ylim(0, lim)
    axes[1].legend(fontsize=8)
    axes[1].set_title("(b) Mean bout duration")

    t_stat, p_val = stats.ttest_rel(hmm_bouts, hsmm_bouts)
    bp = axes[2].boxplot([hmm_bouts, hsmm_bouts], labels=["HMM", "HSMM"],
                         patch_artist=True)
    for patch, colour in zip(bp["boxes"], ["#9ecae1", "#fdae6b"]):
        patch.set_facecolor(colour)
    axes[2].set_ylabel("Mean bout duration (h)")
    axes[2].set_title(f"(c) Paired $t$-test: $p$ = {p_val:.3f}")

    fig.tight_layout()
    save(fig, "hsmm_vs_hmm_comparison.png")

    json.dump(
        {
            "subjects": codes,
            "agreement_mean": float(np.mean(agreements)),
            "agreement_sd": float(np.std(agreements)),
            "hmm_mean_bout_h": float(np.mean(hmm_bouts)),
            "hsmm_mean_bout_h": float(np.mean(hsmm_bouts)),
            "paired_t": float(t_stat),
            "p_value": float(p_val),
            "lambda_by_state": {int(k): float(v) for k, v in summary.items()},
        },
        open(OUTPUTS / "hsmm_comparison.json", "w", encoding="utf-8"),
        indent=2,
    )


def fig_lmm_forest() -> None:
    path = OUTPUTS / "lmm_summaries.txt"
    if not path.exists():
        logger.warning("lmm_summaries.txt missing; skipping forest plot")
        return

    # Blocks are introduced by "MODEL: <name>"; coefficient rows carry a name,
    # estimate, standard error, z, p and the two CI bounds.
    blocks: dict[str, list[str]] = {}
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if "MODEL:" in line:
            current = line.split("MODEL:")[1].strip()
            blocks[current] = []
        elif current is not None:
            blocks[current].append(line)

    models = [k for k in blocks if "sex" in k.lower()][:2]
    if not models:
        logger.warning("no sex models found in lmm_summaries.txt")
        return

    fig, axes = plt.subplots(len(models), 1, figsize=(FIG_W, 3.6 * len(models)))
    if len(models) == 1:
        axes = [axes]

    for ax, name in zip(axes, models):
        rows = []
        for line in blocks[name]:
            parts = line.split()
            if len(parts) < 6 or "sex" not in parts[0].lower():
                continue
            try:
                coef = float(parts[1])
                lo, hi = float(parts[-2]), float(parts[-1])
            except ValueError:
                continue
            rows.append((parts[0].replace("[T.Male]", ""), coef, lo, hi))

        if not rows:
            ax.text(0.5, 0.5, "No sex terms reported", ha="center", va="center")
            ax.set_axis_off()
            continue

        labels = [r[0] for r in rows]
        coefs = np.array([r[1] for r in rows])
        lo = np.array([r[2] for r in rows])
        hi = np.array([r[3] for r in rows])
        y = np.arange(len(rows))
        sig = (lo > 0) | (hi < 0)

        ax.errorbar(coefs, y, xerr=[coefs - lo, hi - coefs], fmt="o",
                    color="#3182bd", ecolor="gray", capsize=3, ms=5)
        if sig.any():
            ax.scatter(coefs[sig], y[sig], color="crimson", zorder=3, s=45,
                       label="CI excludes zero")
            ax.legend(fontsize=8, loc="lower right")
        ax.axvline(0, color="black", ls="--", lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("Coefficient (95% CI)")
        ax.set_title(f"{name}  ($n$ = 15 subjects)")

    fig.suptitle("Sex fixed effects, multi-component cosinor mixed models")
    fig.tight_layout()
    save(fig, "glmm_forest_plots.png")


def main() -> None:
    df = load()
    logger.info(f"{len(df):,} observations, {df['subject_code'].nunique()} subjects")

    fig_posture_validation(df)
    fig_model_selection(df)
    fig_transition_matrix(df)
    fig_dwell_times(df)
    fig_bout_normality(df)
    fig_state_probability_by_hour(df)
    fig_between_subject(df)
    fig_sex_effects(df)
    fig_cosinor(df)
    fig_hsmm(df)
    fig_lmm_forest()

    logger.success(f"figures written to {FIGDIR}")


if __name__ == "__main__":
    main()
