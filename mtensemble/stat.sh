#!/usr/bin/env bash

set -e

. functions
END=9


for i in $(seq 0 $END)
do
    FILE=$OUTPUT_EXPERIMENT/test-${i}-rf.pred
    awk 'BEGIN { FS=",";OFS="\t" } NF {gsub("^[21]:","",$2); gsub("^[21]:","",$3); print $6,"_error_type_","_org_length_","_correction_",$2,$3,$5}' $FILE | ./compute_statistics.pl

done
