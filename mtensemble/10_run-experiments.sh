#!/usr/bin/env bash

set -e

. functions
. local

mkdir -p "${INPUT_EXPERIMENT}"
mkdir -p "${OUTPUT_EXPERIMENT}"

### Run 10-fold CV on the 'largest' data set
#
#

FEATSET=${INPUT_FEATSET}/data.arff
TRAIN_DATA=${INPUT_EXPERIMENT}/data.arff
ln -f -s "${FEATSET}" "${TRAIN_DATA}"
OUTFILE="${OUTPUT_EXPERIMENT}/data-rf"
[ -e "$OUTFILE".eval ] || weka-rf "$TRAIN_DATA" "$OUTFILE"


END=9

for i in $(seq 0 $END)
do
    echo 'fold: '${i}
    mkdir -p ${INPUT_EXPERIMENT}/TMP/TRAIN
    mkdir -p ${INPUT_EXPERIMENT}/TMP/TEST

    cp ${INPUT_FEATSET}/fold${i}.csv ${INPUT_EXPERIMENT}/TMP/TEST
    cd ${INPUT_EXPERIMENT}/TMP/TEST
    convert01 fold${i}.csv test-${i}.arff
    TEST_DATA=${INPUT_EXPERIMENT}/TMP/TEST/test-${i}.arff

    touch ${INPUT_EXPERIMENT}/TMP/TRAIN/train-${i}.csv
    head -1 ${INPUT_FEATSET}/fold${i}.csv > ${INPUT_EXPERIMENT}/TMP/TRAIN/train-${i}.csv
    chmod -R 755 ${INPUT_EXPERIMENT}

    for j in $(seq 0 $END)
    do
        if [[ ${j} -ne ${i} ]]
        then
            echo 'train set: '${j}
            cat ${INPUT_FEATSET}/fold${j}.csv | sed 1d >> ${INPUT_EXPERIMENT}/TMP/TRAIN/train-${i}.csv
        fi
    done
    cd ${INPUT_EXPERIMENT}/TMP/TRAIN
    convert01 train-${i}.csv train-${i}.arff
    TRAIN_DATA=${INPUT_EXPERIMENT}/TMP/TRAIN/train-${i}.arff
    OUT_BASE=$(basename "$TEST_DATA" .arff)
    OUTFILE=${OUTPUT_EXPERIMENT}/${OUT_BASE}-rf
    weka-rf "$TRAIN_DATA" "$OUTFILE" "$TEST_DATA"
#    rm -r ${INPUT_EXPERIMENT}/TMP
done
