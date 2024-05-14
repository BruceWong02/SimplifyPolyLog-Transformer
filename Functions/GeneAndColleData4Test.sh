#!/bin/bash

python FuncGener_general.py
cd ./data/generated/

if [ $(ls -1 tokens_* | wc -l) -eq 200 ]; then
    cd ./merged_files/
    if [ $(ls -1 tokens_* | wc -l) -eq 40 ]; then
        cat tokens_*.csv > tokens_merged.csv
        echo "src,tgt,src_sp,tgt_sp,src_mma,tgt_mma,ns,nt,n_scr" > Test_tokens_finial.csv
        cat tokens_merged.csv >> Test_tokens_finial.csv
        wc -l Test_tokens_finial.csv
        mv Test_tokens_finial.csv ../
    else
        echo "Error: Expected 40 files starting with 'tokens_' but found $(ls -1 tokens_* | wc -l) files."
        exit 1
    fi
else
    echo "Error: Expected 200 files starting with 'tokens_' but found $(ls -1 tokens_* | wc -l) files."
    exit 1
fi



