#!/bin/bash

for sample in ./input/*;do
	echo "$sample"
	sbatch ./run_esmrank_pipeline.sh --input "$sample" -sub
done
