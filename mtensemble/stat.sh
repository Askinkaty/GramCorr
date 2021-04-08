#!/usr/bin/env bash

set -e

. functions
END=9


for i in $(seq 0 $END)
do
    FILE=$OUTPUT_EXPERIMENT/5_guessers-10_folds/0_1_2_3_4/fold${i}.pred
    awk 'BEGIN { FS=",";OFS="\t" } NF {gsub("^[21]:","",$2); gsub("^[21]:","",$3); print $6,"_error_type_","_org_length_","_correction_",$2,$3,$5}' $FILE | ./compute_statistics.pl

done
