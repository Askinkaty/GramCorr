#!/usr/bin/env bash

set -e


MODEL=5_gram
CORPUS=KOKO
#CHAR='_char'
CHAR=''

BASE_DIR=/hltsrv0/a.katinskaia/${CORPUS}
CUR_MODEL_DIR=$BASE_DIR/ALL_REVERTED_MODELS/$MODEL

PATH_TO_PROCESSED=$BASE_DIR/DATA/processed${CHAR}


CORPUS_DIR=/hltsrv2/SMT_TUNING_SYSTEM/CORPUS_REPOSITORY
TUNING_DIR=/hltsrv2/SMT_TUNING_SYSTEM/TUNING_REPOSITORY

TUNE_DIR=$CUR_MODEL_DIR/TUNE-${CORPUS}
MY_MODEL_DIR=$CUR_MODEL_DIR/MODELS
TUNED_MODELS=$CUR_MODEL_DIR/TUNED_MODELS
TMP=$MY_MODEL_DIR/TMP

END=9
#mkdir -p $TUNED_MODELS
#chmod -R 755 $MY_MODEL_DIR
#
#for i in $(seq 0 $END)
#do
#    mkdir -p $TMP
#    cp $TUNE_DIR/fold${i}/train.en $TMP
#    cp $TUNE_DIR/fold${i}/train.de $TMP
#    cp $TUNE_DIR/fold${i}/ActualParameters.cfg $TMP
#    cp $TUNE_DIR/fold${i}/moses.ini $TMP
#    chmod -R 755 $TMP
#    cd $TMP
#    sed -i "s|MODELS-${CORPUS}fold${i}-de-en|$MY_MODEL_DIR/fold${i}/MODELS-${CORPUS}fold${i}-de-en|" moses.ini
#    tar -czvf train-${CORPUS}10fold${i}-de-en.tar.gz train.en train.de ActualParameters.cfg moses.ini
#    cp train-${CORPUS}10fold${i}-de-en.tar.gz $CORPUS_DIR
#    cd $CORPUS_DIR
#    touch train-${CORPUS}10fold${i}-de-en.ready
#    rm -rf $TMP
#done

cd $CORPUS_DIR
for i in $(seq 0 $END)
do
    while ! test -f "train-${CORPUS}5fold${i}-de-en.done"
    do
        sleep 60
        echo "Still waiting"
    done
    mkdir -p $TUNED_MODELS/fold${i}
    mv $TUNING_DIR/TUNING-${CORPUS}5fold${i}-de-en.tar.gz $TUNED_MODELS/fold${i}
    rm -rf $CORPUS_DIR/train-${CORPUS}5fold${i}-de-en.done
    rm -rf $CORPUS_DIR/train-${CORPUS}5fold${i}-de-en.tar.gz
    rm -rf $TUNING_DIR/TUNING-${CORPUS}5fold${i}-de-en.done
    cd $TUNED_MODELS/fold${i}
    tar -xzvf TUNING-${CORPUS}5fold${i}-de-en.tar.gz
    cd $CORPUS_DIR
done
