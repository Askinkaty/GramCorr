#!/usr/bin/env bash

ERRANT_DIR=$HOME/GramCorr/boyd-lowresource/errant
CORPUS=KOKO
DATA=cv
folds=9

TRANSLATED_DIR=$HOME/GramCorr/translated/moses/${CORPUS}/${DATA}

for i in $(seq 0 $folds)
do
    cd $TRANSLATED_DIR/fold${i}
    # create m2 format for reference
    python3 $ERRANT_DIR/scripts/parallel_to_m2.py -orig train.en -cor train.de -out ref_fold${i}_m2
    # create m2 format for hypothesis
    python3 $ERRANT_DIR/scripts/parallel_to_m2.py -orig train.en -cor train.en.trans.de -out hp_fold${i}_m2

    python3 $ERRANT_DIR/scripts/compare_m2.py -hyp hp_fold${i}_m2 -ref ref_fold${i}_m2


done