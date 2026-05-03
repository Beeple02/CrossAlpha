# CrossAlpha On Google Colab

This guide is the fastest path to training CrossAlpha on Google Colab.

## Dependencies

Runtime dependencies:

- `numpy>=1.26`
- `pandas>=2.2`
- `pyarrow>=16.1`
- `scikit-learn>=1.5`
- `lightgbm>=4.5`
- `yfinance>=0.2.54`
- `lxml>=5.2`
- `tomli>=2.0` on Python `<3.11`

Verification dependency:

- `pytest>=8.2`

The repository now includes [requirements-colab.txt](C:\Users\lucia\Documents\Codex\2026-05-03-files-mentioned-by-the-user-crossalpha\requirements-colab.txt), which installs the project and `pytest`.

## Recommended Colab Runtime

- Runtime type: `Python 3`
- Hardware accelerator: `None` is fine
- High-RAM runtime: helpful but optional

## Notebook Setup

Run these cells in order.

### 1. Mount Google Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

### 2. Clone or upload the repo

If the repo is on GitHub:

```bash
%cd /content
!git clone <YOUR_REPO_URL> CrossAlpha
%cd /content/CrossAlpha
```

If you uploaded a zip or folder manually, just `cd` into the repo root.

### 3. Install dependencies

```bash
%cd /content/CrossAlpha
!python -m pip install --upgrade pip
!pip install -r requirements-colab.txt
```

### 4. Set a descriptive user-agent

This helps avoid `403` responses from Wikipedia and keeps SEC requests compliant.

```python
import os
os.environ["CROSSALPHA_USER_AGENT"] = "CrossAlpha Research your_email@example.com"
```

### 5. Use the Colab config

Use [configs/colab.toml](C:\Users\lucia\Documents\Codex\2026-05-03-files-mentioned-by-the-user-crossalpha\configs\colab.toml). It writes artifacts to:

```text
/content/drive/MyDrive/CrossAlphaArtifacts/
```

That means your downloaded data, trained models, logs, and reports survive notebook restarts.

## Training Commands

### Full end-to-end pipeline

```bash
!crossalpha run-all --config configs/colab.toml
```

### Stage-by-stage version

```bash
!crossalpha ingest --config configs/colab.toml
!crossalpha features --config configs/colab.toml
!crossalpha labels --config configs/colab.toml
!crossalpha validate --config configs/colab.toml
!crossalpha train --config configs/colab.toml
!crossalpha recommend --config configs/colab.toml --latest-only
!crossalpha backtest --config configs/colab.toml
```

## What “Finished Training” Looks Like

For validation, you should see log lines like:

```text
Finished 2w split_3 ranker | NDCG@20=...
Validation complete.
```

For final model fitting, you should see:

```text
Finished horizon 2m final training. Baseline -> ...
Completed final model training for 5 horizons.
```

Those are the clearest success markers in notebook output.

You can also confirm completion by checking that these files exist:

- `/content/drive/MyDrive/CrossAlphaArtifacts/data/models/1d_ranker.pkl`
- `/content/drive/MyDrive/CrossAlphaArtifacts/data/models/1w_ranker.pkl`
- `/content/drive/MyDrive/CrossAlphaArtifacts/data/models/2w_ranker.pkl`
- `/content/drive/MyDrive/CrossAlphaArtifacts/data/models/1m_ranker.pkl`
- `/content/drive/MyDrive/CrossAlphaArtifacts/data/models/2m_ranker.pkl`
- `/content/drive/MyDrive/CrossAlphaArtifacts/reports/output/final_model_registry.json`
- `/content/drive/MyDrive/CrossAlphaArtifacts/reports/output/validation_summary.json`

## Quick Verification In Colab

After training, run:

```bash
!ls /content/drive/MyDrive/CrossAlphaArtifacts/data/models
!cat /content/drive/MyDrive/CrossAlphaArtifacts/reports/output/final_model_registry.json
```

## If You Only Want To Train Models

If ingestion and feature generation already ran before, you can restart at:

```bash
!crossalpha train --config configs/colab.toml
```

If you want fresh validation metrics before training final models:

```bash
!crossalpha validate --config configs/colab.toml
!crossalpha train --config configs/colab.toml
```

## Common Notes

- The first full run can take a while because it downloads market data, earnings, fundamentals, and benchmark series.
- Free-source fundamentals and historical index membership are the slowest / least reliable parts of the pipeline.
- `NOT_ENOUGH_DATA` is expected for some names and dates. That is part of the design, not a crash.
- If you want to keep artifacts inside the notebook filesystem instead of Drive, use [configs/base.toml](C:\Users\lucia\Documents\Codex\2026-05-03-files-mentioned-by-the-user-crossalpha\configs\base.toml) instead.
