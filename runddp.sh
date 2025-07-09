#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export WORLD_SIZE=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
dora run -d model=htdemucs \
    dset.musdb=/mnt/data/xxn/data/musdbhq \
    +precision=fp16 \
    +dset.samples=100 \
    +segment=7 \
    batch_size=16 \
    +augment.group_size=1 \
    +augment.disabled=true
