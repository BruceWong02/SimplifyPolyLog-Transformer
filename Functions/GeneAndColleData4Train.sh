#!/bin/bash

python FuncGener_general.py
cd ./data/generated/

if [ $(ls -1 tokens_* | wc -l) -eq 250 ]; then
    cd ./merged_files/
    if [ $(ls -1 tokens_* | wc -l) -eq 50 ]; then
        cat tokens_*.csv > tokens_merged.csv
        echo "src,tgt,src_sp,tgt_sp,src_mma,tgt_mma,ns,nt,n_scr" > Train_tokens_finial.csv
        cat tokens_merged.csv >> Train_tokens_finial.csv
        wc -l Train_tokens_finial.csv
        mv Train_tokens_finial.csv ../
    else
        echo "Error: Expected 50 files starting with 'tokens_' but found $(ls -1 tokens_* | wc -l) files."
        exit 1
    fi
else
    echo "Error: Expected 250 files starting with 'tokens_' but found $(ls -1 tokens_* | wc -l) files."
    exit 1
fi



