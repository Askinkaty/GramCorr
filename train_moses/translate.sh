#!/usr/bin/env bash

set -e

KOKO_DIR=/hltsrv0/a.katinskaia/KOKO


CORPUS_DIR=/hltsrv2/SMT_TRANSLATION_SYSTEM/CORPUS_REPOSITORY
TRANSLATION_DIR=/hltsrv2/SMT_TRANSLATION_SYSTEM/TRANSLATION_REPOSITORY
TEST_DIR=$KOKO_DIR/TEST-KOKO
TUNED_MODELS=$KOKO_DIR/TUNED_MODELS
TRANSLATED=$KOKO_DIR/TRANSLATED
TMP=$KOKO_DIR/ready_translate


PATH_TO_PROCESSED=$KOKO_DIR/processed


END=9
mkdir -p $TRANSLATED

for i in $(seq 0 $END)
do
    cp $PATH_TO_PROCESSED/processed_fold${i}/PROCESSED-KOKOfold${i}-de/train.de $TEST_DIR/fold${i}
    cd $TUNED_MODELS/fold${i}
    tar -xzvf TUNING-KOKOfold${i}-en-de.tar.gz
    mkdir -p $TMP
    cp $TEST_DIR/fold${i}/train.en $TMP
    cp $TEST_DIR/fold${i}/ActualParameters.cfg $TMP
    cp $TUNED_MODELS/fold${i}/TUNING-KOKOfold${i}-en-de/moses.ini $TMP
    cd $TMP
    tar -czvf train-KOKOfold${i}-en-de.tar.gz train.en train.en ActualParameters.cfg moses.ini
    cp train-KOKOfold${i}-en-de.tar.gz $CORPUS_DIR
    rm -rf $TMP
    cd $CORPUS_DIR
    touch train-KOKOfold${i}-en-de.ready
done

cd $CORPUS_DIR
for i in $(seq 0 $END)
do
    while ! test -f "$CORPUS_DIR/train-KOKOfold${i}-en-de.done"
    do
        sleep 60
        echo "Still waiting"
    done
    mkdir -p $TRANSLATED/fold${i}
    mv $TRANSLATION_DIR/TRANSLATION-KOKOfold${i}-en-de.tar.gz $TRANSLATED/fold${i}
    rm -f $TRANSLATION_DIR/TRANSLATION-KOKOfold${i}-en-de.done
    rm -f $CORPUS_DIR/train-KOKOfold${i}-en-de.tar.gz
    rm -f $CORPUS_DIR/train-KOKOfold${i}-en-de.done
    tar -xzvf $TRANSLATED/fold${i}/TRANSLATION-KOKOfold${i}-en-de.tar.gz
    cd $CORPUS_DIR
done