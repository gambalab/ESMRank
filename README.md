# ESMRank  
**A learn-to-rank approach for protein variant effect prediction**

ESMRank is a pipeline for ranking protein variants according to their predicted functional impact.  
The method frames variant effect prediction as a **learning-to-rank problem**, leveraging protein language models and sequence-derived features to prioritize mutations based on fitness, activity, or stability.

---

## Installation

### Requirements
- Python **3.10** (recommended)
- Linux / macOS environment
- `bash`

### Setup instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/j3rk0/ESMRank.git
   ```

2. **Move into the repository directory**
   ```bash
   cd ESMRank
   ```

3. **Create a virtual environment**
   ```bash
   python -m venv ESMRank_venv
   ```

4. **Activate the virtual environment**
   ```bash
   source ESMRank_venv/bin/activate
   ```

5. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

6. **Download pretrained models**
   
   Download `ESMRank_models.tar.gz` from:

   ```
   https://zenodo.org/records/18773439
   ```

   Then extract it in the repository root:
   ```bash
   tar -xvf ESMRank_models.tar.gz
   ```
7. move the ```models``` directory into the repository root:
   ```bash
   mv ESMRank/model ./
   ```
   
   

---

## Usage

The main pipeline can be executed via:

```bash
run_esmrank_pipeline.sh [-h] --input INPUT [-csv] [-sub] [-indel] [-alanines]
```

---

## Input modes

ESMRank supports **two input modes**: CSV mode and FASTA mode.

---

### CSV mode

In CSV mode, the input file must be a CSV containing **three mandatory columns**:

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
run_esmrank_pipeline.sh --input variants.csv -csv
```

---

### FASTA mode

In FASTA mode, the input must be a **single wild-type protein sequence** in FASTA format.

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
run_esmrank_pipeline.sh --input protein.fasta -sub
```
generating all single single indel

```bash
run_esmrank_pipeline.sh --input protein.fasta -indel
```
generating all missenses and single indel

```bash
run_esmrank_pipeline.sh --input protein.fasta -sub -indel
```
generate alanine scanning ( alanines substitution only ) 

```bash
run_esmrank_pipeline.sh --input protein.fasta -alanines
```

---

## Command-line arguments

| Argument | Description |
|--------|-------------|
| `-h` | Show help message |
| `--input` | Path to input file (CSV or FASTA) |
| `-csv` | Enable CSV input mode (default: FASTA mode) |
| `-sub` | Generate all possible missense substitutions |
| `-indel` | Generate all possible insertions and deletions |

---

## Notes

- CSV and FASTA modes are **mutually exclusive**.
- In FASTA mode, at least one of `-sub` or `-indel` must be specified.
- Runtime and memory usage scale with protein length and number of generated variants.

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

