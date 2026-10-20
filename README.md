# ESMRank  
**A learn-to-rank approach for protein variant effect prediction**

ESMRank is a pipeline for ranking protein variants according to their predicted functional impact.  
The method frames variant effect prediction as a **learning-to-rank problem**, leveraging protein language models and sequence-derived features to prioritize mutations based on fitness, activity, or stability.

---

## Installation

### Requirements
- Python **3.10+**
- Linux / macOS
- A CUDA GPU (the ESM-2 feature step needs one)

### Install with pip

Install a `torch` build that matches your CUDA setup first (see https://pytorch.org), then:

```bash
pip install git+https://github.com/gambalab/ESMRank.git
```

This provides the `esmrank` command (and `python -m esmrank`).

### Download the pretrained models

The package does not bundle the models (~830 MB, hosted on Zenodo: https://zenodo.org/records/18773439). Fetch them once with:

```bash
esmrank download-models
```

They are stored in `~/.cache/esmrank/model` (or `--dest DIR`). If the installation fails the downloaded archive is kept, so re-running does not download it again; an archive you already have can be used with `esmrank download-models --archive ESMRank.tar.gz`. At run time the model directory is looked up in this order: `--model-dir`, `$ESMRANK_MODEL_DIR`, `./model`, then the cache directory above.

### Install from source (development)

```bash
git clone https://github.com/gambalab/ESMRank.git
cd ESMRank
python -m venv ESMRank_venv && source ESMRank_venv/bin/activate
pip install -e ".[test]"
esmrank download-models
pytest
```

`requirements.txt` still lists the exact versions the pipeline was developed against; `pyproject.toml` declares compatible ranges.

---

## Usage

### From Python

```python
import pandas as pd
import esmrank

# make sure the pretrained files are available (downloads them only if missing)
model_dir = esmrank.ensure_models()

# a protein sequence instead of a FASTA file
seq = "MSEQNNTEMTFQIQRIYTKDISFEAPNAPHVFQKDWQPEVKLDLDTASSQLADDVYEVVLRVTVTASLG"
df = esmrank.rank_variants(seq, sub=True)                    # all missense variants
df = esmrank.rank_variants(sequence=seq, alanines=True)      # explicit form, any length

# a DataFrame instead of a CSV file (columns hgvsp, seq_wt, seq_mu; extra columns are kept)
variants = pd.DataFrame({"hgvsp": ["p.Q12V"], "seq_wt": [seq], "seq_mu": [seq[:11] + "V" + seq[12:]]})
df = esmrank.rank_variants(variants)

# files work as well
df = esmrank.rank_variants("protein.fasta", sub=True, model_dir=model_dir)
df = esmrank.rank_variants("variants.csv", csv=True)

# download the models first if they are not there yet
df = esmrank.rank_variants(seq, alanines=True, auto_download=True)
```

`rank_variants` returns a DataFrame with columns `ids` and `prediction`.

How the input is interpreted:

| You pass | Treated as |
|----------|------------|
| a `pandas.DataFrame` | table of variants (same as CSV mode); `sub` / `indel` / `alanines` do not apply |
| a string that is an existing file | FASTA file, or CSV file with `csv=True` |
| a string starting with `>` | text of a single-record FASTA |
| an all-uppercase string of at least 10 letters that is not an existing file | protein sequence |
| `sequence="..."` | protein sequence, explicitly (any length, any case) |

A sequence needs at least one of `sub`, `indel`, `alanines`, and may only contain the 20 standard amino acids. Any other string that is not a file raises `FileNotFoundError`, so a mistyped file name is never read as a sequence (use `sequence=` for short ones). Results are still written to `<outdir>/<name>/`: `name` defaults to the file name, the FASTA header, or a short hash of the sequence / DataFrame, and can be set with `name="..."`. The command line only accepts files.

| Function | What it does |
|----------|--------------|
| `esmrank.ensure_models(model_dir=None)` | Returns the model directory; downloads the files only if they are missing (into `model_dir`, `$ESMRANK_MODEL_DIR`, or `~/.cache/esmrank/model`) |
| `esmrank.download_models(dest=None, force=False, verify=True)` | Always downloads (unless already present and not `force`) |
| `esmrank.find_model_dir(explicit=None)` | Locates existing model files, raises `FileNotFoundError` if none |
| `esmrank.rank_variants(data, ...)` | Runs the pipeline and returns the predictions as a DataFrame |
| `esmrank.run_pipeline(data, ...)` | Runs the pipeline and returns the output directory (also has `profile.tsv` / `profile.png`) |

`rank_variants` and `run_pipeline` take the same options as the CLI: `csv`, `sub`, `indel`, `alanines`, `outdir`, `model_dir`, `esm_model`, `batch_size`, `gpu_ids`, plus `sequence`, `name` and `auto_download`. On the command line the equivalent of `auto_download` is `esmrank run --download ...`.

### From command line

```bash
esmrank run --input INPUT [-csv] [-sub] [-indel] [-alanines]
            [--outdir DIR] [--model-dir DIR] [--download] [--gpus 0,1] [--batch-size N] [--esm-model NAME]
```

Results are written to `<outdir>/<input file name>/` (default `output/<name>/`): `predictions.csv`, plus `profile.tsv` and `profile.png` in `-sub` / `-alanines` mode.

---

## Input modes

ESMRank supports **two input modes**: CSV mode and FASTA mode.

From the command line the input is always a file. From Python the input can also be given in memory: a **pandas DataFrame** instead of the CSV file, and a **protein sequence string** instead of the FASTA file (see [From Python](#from-python) for all the details).

---

### CSV mode

In CSV mode, the input must be a CSV file (or, from Python, a pandas DataFrame) containing **three mandatory columns**:

| Column name | Description |
|------------|-------------|
| `hgvsp` | Variant identifier |
| `seq_wt` | Wild-type protein sequence |
| `seq_mu` | Mutant protein sequence |

Example:
```csv
hgvsp,seq_wt,seq_mu
p.A123V,MSEQNNTEMTFQIQRIYTKDISFEAPNAPHVFQ...,MSEQNNTEMTFQIQRIYTKDISFEVPNAPHVFQ...
```

To run in CSV mode:
```bash
esmrank run --input variants.csv -csv
```

From Python you can pass a DataFrame directly, without writing a CSV (extra columns are kept):
```python
import pandas as pd
import esmrank

variants = pd.read_csv("variants.csv")      # or any DataFrame with the three columns above
df = esmrank.rank_variants(variants)        # DataFrame with columns `ids` and `prediction`
```

---

### FASTA mode

In FASTA mode, the input must be a **single wild-type protein sequence** in FASTA format (or, from Python, the sequence itself as a string).

The pipeline will automatically generate variants based on the selected mutation types:

- `-sub` : generate all possible **missense substitutions**
- `-indel` : generate all possible **insertions and deletions**
- `-alanines`: mutually exclusive to sub and indel, generate only alanines substitutions for each aminoacid

Example FASTA:
```fasta
>protein_X
MSEQNNTEMTFQIQRIYTKDISFEAPNAPHVFQ...
```

Example execution:

generating all missenses variants:

```bash
esmrank run --input protein.fasta -sub
```
generating all single single indel

```bash
esmrank run --input protein.fasta -indel
```
generating all missenses and single indel

```bash
esmrank run --input protein.fasta -sub -indel
```
generate alanine scanning ( alanines substitution only ) 

```bash
esmrank run --input protein.fasta -alanines
```

From Python you can pass the sequence as a string instead of a FASTA file; the same `sub`, `indel` and `alanines` options apply:
```python
import esmrank

seq = "MSEQNNTEMTFQIQRIYTKDISFEAPNAPHVFQKDWQPEVKLDLDTASSQLADDVYEVVLRVTVTASLG"
df = esmrank.rank_variants(seq, sub=True)                  # all missense variants
df = esmrank.rank_variants(sequence=seq, alanines=True)    # alanine scanning
```

---

## Command-line arguments

| Argument | Description |
|--------|-------------|
| `-h` | Show help message |
| `--input`, `-i` | Path to input file (CSV or FASTA) |
| `-csv` | Enable CSV input mode (default: FASTA mode) |
| `-sub` | Generate all possible missense substitutions |
| `-indel` | Generate all possible insertions and deletions |
| `-alanines` | Generate alanine substitutions for each position |
| `--outdir` | Output root directory (default: `output`) |
| `--model-dir` | Directory with the pretrained model files |
| `--download` | Download the model files first if they are not already present |
| `--gpus` | Comma-separated CUDA device ids (default: all) |
| `--batch-size` | Worker batch size for the ESM step (default: 128) |
| `--esm-model` | ESM-2 checkpoint name (default: `esm2_t33_650M_UR50D`) |

The command line only accepts files. Sequence strings and DataFrames are available from Python, see [From Python](#from-python).

The environment variables `ESM_GPU_IDS`, `ESM_WORKER_BATCH_SIZE`, `ESM_MODEL_NAME` and `ESM_CHECKPOINT_DIR` are still honoured.

---

## Notes

- CSV and FASTA modes are **mutually exclusive**.
- In FASTA mode, at least one of `-sub`, `-indel` or `-alanines` must be specified.
- Runtime and memory usage scale with protein length and number of generated variants.

---

## License

- **Source code**: Apache License 2.0, see [`LICENSE`](LICENSE).
- **Pretrained model parameters** (downloaded with `esmrank download-models`): ESMRank Model Parameters Non-Commercial Research License, see [`LICENSE_WEIGHTS.md`](LICENSE_WEIGHTS.md). They may be used for non-commercial research and educational purposes only; commercial use requires written permission from the authors.
- Third-party resources (Pfam-A HMMs, ESM-2 weights) keep their own licenses.

---

## Citation

If you use ESMRank in your research, please cite:

```bibtex
@article{esmrank,
  title={ESMRank reveals a transferable axis of protein mutational constraint from overlapping variant effect assays},
  author={Riccardo Arnese, Gennaro Gambardella},
  journal={TBD},
  year={TBD}
}
```
