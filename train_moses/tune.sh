#!/usr/bin/env bash

set -e

KOKO_DIR=/hltsrv0/a.katinskaia/KOKO


CORPUS_DIR=/hltsrv2/SMT_TUNING_SYSTEM/CORPUS_REPOSITORY
TUNING_DIR=/hltsrv2/SMT_TUNING_SYSTEM/TUNING_REPOSITORY

TUNE_DIR=$KOKO_DIR/TUNE-KOKO
MY_MODEL_DIR=$KOKO_DIR/MODELS
TUNED_MODELS=$KOKO_DIR/TUNED_MODELS
TMP=$MY_MODEL_DIR/TMP

END=9
mkdir -p $TUNED_MODELS
chmod -R 755 $MY_MODEL_DIR

for i in $(seq 0 $END)
do
    mkdir -p $TMP
    cp $TUNE_DIR/fold${i}/train.en $TMP
    cp $TUNE_DIR/fold${i}/train.de $TMP
    cp $TUNE_DIR/fold${i}/ActualParameters.cfg $TMP
    cp $TUNE_DIR/fold${i}/moses.ini $TMP
    chmod -R 755 $TMP
    cd $TMP
    sed -i "s|MODELS-KOKOfold${i}-en-de|$MY_MODEL_DIR/fold${i}/MODELS-KOKOfold${i}-en-de|" moses.ini
    tar -czvf train-KOKOfold${i}-en-de.tar.gz train.en train.de ActualParameters.cfg moses.ini
    cp train-KOKOfold${i}-en-de.tar.gz $CORPUS_DIR
    cd $CORPUS_DIR
    touch train-KOKOfold${i}-en-de.ready
    rm -rf $TMP
done

cd $CORPUS_DIR
for i in $(seq 0 $END)
do
    while ! test -f "$CORPUS_DIR/train-KOKOfold${i}-en-de.done"
    do
        sleep 60
        echo "Still waiting"
    done
    mkdir $TUNED_MODELS/fold${i}
    mv $TUNING_DIR/TUNING-KOKOfold${i}-en-de.tar.gz $TUNED_MODELS/fold${i}
    tar -xzvf $TUNED_MODELS/fold${i}/TUNING-KOKOfold${i}-en-de.tar.gz
    cd $CORPUS_DIR
done