#!/bin/bash
#SBATCH -N 1                # 1 node
#SBATCH -J ESMRank
#SBATCH --ntasks-per-node=1 # 4 tasks out of 32
#SBATCH --gres=gpu:1        # 4 gpus per node out of 4
cd "$(dirname "$0")"
echo ""
echo "   _____ ________  _________            _ "
echo "  |  ___/  ___|  \/  || ___ \          | | "
echo "  | |__ \ \`--.| .  . || |_/ /__ _ _ __ | | __"
echo "  |  __| \`--. \ |\/| ||    // _\` | '_ \| |/ /"
echo "  | |___/\__/ / |  | || |\ \ (_| | | | |   <  "
echo "  \____/\____/\_|  |_/\_| \_\__,_|_| |_|_|\_\  "
echo ""

if [[ "$1" == "-h" ]]; then
	echo "----ESMRank: A Learn-To-Rank-based VEP tool----"
	echo "this script will run the ESMRank tool on a given"
	echo "fasta sequence OR on a given csv file"
	echo ""
	echo "----Usage:"
	echo "run_esmrank_pipeline.sh [-h] [--input INPUT] [-csv] [-sub] [-indel]"
	echo ""
	echo "if csv parameters is used, input file must be a csv"
	echo "with three columns hgvsp, seq_wt and seq_mu"
	echo "respectively for variant identifier, wild type"
	echo "sequence and mutant sequence."
	echo "if 'sub' and/or 'indel' parameters are used, "
	echo "the input file must be a fasta and the script"
	echo "will generate the selected types of variants by itself"
	echo ""
	echo "----Parameters:"
	echo "-h: show this help"
	echo "--input ./path/to/input: input file path"
	echo "-csv: read variants directly from csv"
	echo "-sub: generate all possible missense from fasta file"
	echo "-indel: generate all possible indel from fasta file"
	echo ""
	exit 0
fi

input_file=$(echo "$@" | sed -n 's/.*\(--input\|-i\) \([^ ]*\).*/\2/p')


export ESM_MODEL_NAME="esm2_t33_650M_UR50D"
export ESM_WORKER_BATCH_SIZE="128"
export PYTHONPATH="${PYTHONPATH}:./lib"
export SAMPLE_FILE=$(basename -- "$input_file")
export SAMPLE_NAME="${SAMPLE_FILE%.*}"
export ESM_CHECKPOINT_DIR="output/${SAMPLE_NAME}/checkpoints"

mkdir -p "output/${SAMPLE_NAME}"

echo 'generating variants'
python3 scripts/generate_dms.py "$@"

echo 'shallow features extraction'
python3 scripts/feature_extraction.py

echo 'deep features extraction'
python3 scripts/esm_features.py

echo 'merge features'
python3 scripts/merge_features.py

echo 'make predictions'
python3 scripts/predict.py

echo 'building profile'
python3 scripts/scores2profile.py "$@"
