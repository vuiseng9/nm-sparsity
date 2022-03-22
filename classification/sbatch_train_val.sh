#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

CONDAROOT=/nvme1/vchua/miniconda3
CONDAENV=nm-sparsity
WORKDIR=/nvme1/vchua/dev/nm-sparsity/nm-sparsity/classification

NOW=$(date +"%Y%m%d_%H%M%S")

cmd="
python -m torch.distributed.launch \
    --nproc_per_node=4 train_imagenet.py \
    --config configs/cfg_rn50_4oo8.yaml
"

# 2>&1|tee train-$NOW.log

if [[ $1 == "local" ]]; then
    echo "${cmd}" > $OUTDIR/run.log
    echo "### End of CMD ---" >> $OUTDIR/run.log
    cmd="nohup ${cmd}"
    eval $cmd >> $OUTDIR/run.log 2>&1 &
    echo "logpath: $OUTDIR/run.log"
elif [[ $1 == "dryrun" ]]; then
    echo "[INFO: dryrun, add --max_steps 25 to cli"
    cmd="${cmd} --max_steps 25"
    echo "${cmd}" > $OUTDIR/dryrun.log
    echo "### End of CMD ---" >> $OUTDIR/dryrun.log
    eval $cmd >> $OUTDIR/dryrun.log 2>&1 &
    echo "logpath: $OUTDIR/dryrun.log"
else
    source $CONDAROOT/etc/profile.d/conda.sh
    conda activate ${CONDAENV}
    cd $WORKDIR
    eval $cmd
fi

