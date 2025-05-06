# ESMRank: A learn-to-rank approach for protei variant effect prediction

### Installation instruction

1) clone the repository
2) build a virtual enviroment
3) install requirements
4) download Pfam-A.hmm.gz from https://www.ebi.ac.uk/interpro/download/pfam/ and unzip it into lib directory
5) download trained model and put it in models directory

### Run instructions

```bash
run_esmrank_pipeline.sh [-h] [--input INPUT] [-csv] [-sub] [-indel]
```

#### Description:

esmrank could be ran in csv and in fasta mode. csv mode input file should be a csv fire containing three mandatory columns:
- hgvsp - containing an identifier for each variation
- seq_wt - wild type protein sequence
- seq_mu - mutant protein sequence

if ran in fasta mode, you must specify sub and/or indel. In this way the script will generate all the possible mutation for the given sequence

### Execution Parameters:
1) -h will show the help
2) --input ./path/to/input: input file path
3) -csv: run in csv mode ( default is fasta mode )
4) -sub: generate all possible missense from fasta file
5) -indel: generate all possible indel from fasta file
