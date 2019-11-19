#!/usr/bin/env bash

set -e


MODEL=5_gram
CORPUS=KOKO

BASE_DIR=/hltsrv0/a.katinskaia/${CORPUS}
CUR_MODEL_DIR=$BASE_DIR/ALL_MODELS/$MODEL


CORPUS_DIR=/hltsrv2/SMT_TRANSLATION_SYSTEM/CORPUS_REPOSITORY
TRANSLATION_DIR=/hltsrv2/SMT_TRANSLATION_SYSTEM/TRANSLATION_REPOSITORY


MY_MODEL_DIR=$CUR_MODEL_DIR/MODELS

TEST_DIR=$CUR_MODEL_DIR/TEST-${CORPUS}
TUNED_MODELS=$CUR_MODEL_DIR/TUNED_MODELS
TRANSLATED=$CUR_MODEL_DIR/TRANSLATED-${CORPUS}
TMP=$MY_MODEL_DIR/TMP


END=9
mkdir -p $TRANSLATED

for i in $(seq 0 $END)
do
    cd $TUNED_MODELS/fold${i}
    tar -xzvf TUNING-${CORPUS}5fold${i}-en-de.tar.gz
    mkdir -p $TMP
    cp $TEST_DIR/fold${i}/train.en $TMP
    cp $TEST_DIR/fold${i}/ActualParameters.cfg $TMP
    cp $TUNED_MODELS/fold${i}/TUNING-${CORPUS}5fold${i}-en-de/moses.ini $TMP
    cd $TMP
    tar -czvf train-${CORPUS}fold${i}-en-de.tar.gz train.en train.en ActualParameters.cfg moses.ini
    cp train-${CORPUS}fold${i}-en-de.tar.gz $CORPUS_DIR
    rm -rf $TMP
    cd $CORPUS_DIR
    touch train-${CORPUS}fold${i}-en-de.ready
done

cd $CORPUS_DIR
for i in $(seq 0 $END)
do
    while ! test -f "$CORPUS_DIR/train-${CORPUS}fold${i}-en-de.done"
    do
        sleep 60
        echo "Still waiting"
    done
    mkdir -p $TRANSLATED/fold${i}
    mv $TRANSLATION_DIR/TRANSLATION-${CORPUS}fold${i}-en-de.tar.gz $TRANSLATED/fold${i}
    rm -f $TRANSLATION_DIR/TRANSLATION-${CORPUS}fold${i}-en-de.done
    rm -f $CORPUS_DIR/train-${CORPUS}fold${i}-en-de.tar.gz
    rm -f $CORPUS_DIR/train-${CORPUS}fold${i}-en-de.done
    tar -xzvf $TRANSLATED/fold${i}/TRANSLATION-${CORPUS}fold${i}-en-de.tar.gz
    cd $CORPUS_DIR
done