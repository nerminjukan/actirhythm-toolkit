# ActiRhythm Toolkit

ActiRhythm Toolkit is a command-line workflow for analysing accelerometer activity data and characterising behavioural and circadian rhythms. It provides preprocessing, hidden Markov model state estimation, downstream rhythm analysis, and optional machine-learning and GLMM workflows.

This repository contains source code, notebooks and published figures only. **No subject data is distributed**: the maned wolf biologger recordings analysed in the accompanying thesis belong to the data owners and are not redistributable. Supply your own input files under `data/` to run the workflow.

## Repository layout

```
src/         importable package (actirhythm CLI and analysis modules)
scripts/     pipeline entry points, figure generators and reporting helpers
notebooks/   exploratory notebooks (outputs stripped)
tests/       unit tests
figures/     figures published in the thesis
docs/         pipeline description and methodological notes
data/        input data (git-ignored, placeholders only)
outputs/     run artefacts (git-ignored)
```

## Installation

### Install from PyPI (recommended)

```bash
python -m pip install actirhythm-toolkit
```

Optional extras from PyPI:

```bash
python -m pip install "actirhythm-toolkit[ml]"
python -m pip install "actirhythm-toolkit[glmm]"
python -m pip install "actirhythm-toolkit[notebooks]"
```

PyPI project page:

https://pypi.org/project/actirhythm-toolkit/

### Optional capabilities

```bash
python -m pip install "actirhythm-toolkit[ml]"
python -m pip install "actirhythm-toolkit[glmm]"
python -m pip install "actirhythm-toolkit[notebooks]"
```

The `glmm` extra also requires a local R installation and compatible R packages. Check your environment with:

```bash
actirhythm glmm-doctor
```

## Quick Start

Run the complete workflow from a project directory containing your input data and optional `config.yaml`:

```bash
actirhythm
```

Inspect the planned inputs and outputs before running:

```bash
actirhythm --dry-run
```

Run a named analysis version:

```bash
actirhythm full --run-version v1
```

## Commands

```bash
actirhythm full
actirhythm preprocess
actirhythm analytics
actirhythm glmm-doctor
```

Run `actirhythm --help` for all configuration, data-path, output-path, and resume options.

## Input Data

The workflow accepts accelerometer CSV data with a timestamp, subject identifier, activity measure, and optional axis or posture measurements. Use `--data-revised-dir` to specify an input directory and `--raw-data-file` to provide a fallback CSV file.

### Choosing an activity signal

Set the analysis signal with `--activity-signal` (default `ActMindata`).

Some loggers export only the **average** of each accelerometer axis per epoch. The
magnitude of a mean acceleration vector is dominated by gravity, so it describes body
**orientation**, not movement intensity: it is longest when the animal holds a single
posture and shortens as orientation varies within the epoch. Such a magnitude can
therefore be _negatively_ correlated with real activity and must not be used as an
intensity index.

The toolkit still computes `activity_xy` and `activity_xyz` from the axes, but treats them
as posture descriptors. A genuine intensity metric (ODBA/VeDBA) requires within-epoch
variance, which averaged exports do not retain. Where a firmware activity count such as
`ActMindata` is available, prefer it.

Posture descriptors remain useful for **validating** decoded states: resting states should
show a higher and less variable posture magnitude than active ones. The preprocessing
stage writes this check to `posture_state_validation.csv`.

### Deployment windows

Records collected after a device is removed cannot be identified from the signal alone —
a retrieved logger still registers varying posture and occasional activity while it is
being transported, which is indistinguishable from genuine low-activity behaviour.
Documented removal times are required.

Because removal times are study metadata that identify individual animals, none are
shipped with this repository. Place your own `deployment_windows.csv` in the project
directory and call `qc.load_deployment_windows()`:

```csv
subject,end_timestamp,reason
A,2024-04-02 00:00,battery failure before explant
B,2023-07-07 10:15,documented device removal
```

Alternatively, pass a mapping directly to `qc.apply_deployment_windows()`. Without a
window file, post-removal records are treated as valid observations.

### Subject anonymisation

`qc.assign_subject_codes()` replaces subject names with codes (`S01`, `S02`, …) ordered by
first observation, and the preprocessing stage writes the mapping to
`subject_code_mapping.csv`. Treat that file as confidential — it is the re-identification
key. It is not required to reproduce any result.

### Dwell-time filtering

`--min-dwell` sets the minimum bout length in bins for the post-decoding filter
(default `2`; at 15-minute sampling that is 30 minutes).

This filter suppresses decoder flicker but also alters the decoded sequence, so transition
probabilities and dwell-time summaries describe the filtered series. Choose the threshold
against your own data rather than by convention: a value above the unfiltered median bout
length discards a large share of genuine bouts. Each run writes
`dwell_time_sensitivity.csv` comparing 15-, 30- and 60-minute thresholds so the choice can
be justified.

## Development

1. Clone the repository:

```bash
git clone https://github.com/nerminjukan/actirhythm-toolkit.git
cd actirhythm-toolkit
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install in editable mode:

```bash
python -m pip install -e .
```

Development extras:

```bash
python -m pip install -e .[dev]
python -m pip install -e .[ml]
python -m pip install -e .[glmm]
python -m pip install -e .[notebooks]
```

## Source

Source code and issue tracking: https://github.com/nerminjukan/actirhythm-toolkit

## License

GNU General Public License, version 3 or later. See [LICENSE](LICENSE).
