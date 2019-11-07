#!/usr/bin/env bash

set -e

HOME_DIR=/hltsrv0/a.katinskaia
PATH_TO_DATASET=$HOME_DIR/KOKO/cv
PATH_TO_PREPARED_DATA=$HOME_DIR/KOKO/prepared
PATH_TO_PROCESSED=$HOME_DIR/KOKO/processed
TMP=$HOME_DIR/KOKO/tmp

END=9

CORPUS_DIR=/hltsrv2/QT21_PROCESSING_SYSTEM/CORPUS_REPOSITORY
PROCESSED_DIR=/hltsrv2/QT21_PROCESSING_SYSTEM/PROCESSED_REPOSITORY

rm -rf $PATH_TO_PREPARED_DATA

for i in $(seq 7 $END)
do
    echo ${i}
    mkdir -p $PATH_TO_PREPARED_DATA/fold${i}
    cp $PATH_TO_DATASET/fold_${i}_source.txt $PATH_TO_PREPARED_DATA/fold${i}/train.en
    cp $PATH_TO_DATASET/fold_${i}_target.txt $PATH_TO_PREPARED_DATA/fold${i}/train.de

    for lang in en de
    do
        chmod 755 $PATH_TO_PREPARED_DATA/fold${i}/train.${lang}
        mkdir -p $TMP
        cp $PATH_TO_PREPARED_DATA/fold${i}/train.${lang} $TMP
        cp $HOME_DIR/KOKO/ActualParameters.cfg $TMP
        cd $TMP
        tar -czvf train-KOKOfold${i}-${lang}.tar.gz train.${lang} ActualParameters.cfg
        cp train-KOKOfold${i}-${lang}.tar.gz $CORPUS_DIR
        cd $CORPUS_DIR
        touch train-KOKOfold${i}-${lang}.ready
        while ! test -f "$CORPUS_DIR/train-KOKOfold${i}-${lang}.done"
        do
            sleep 60
            echo "Still waiting"
        done
        rm -rf $TMP
        mkdir -p $PATH_TO_PROCESSED/processed_fold${i}
        mv $PROCESSED_DIR/PROCESSED-KOKOfold${i}-${lang}.tar.gz $PATH_TO_PROCESSED/processed_fold${i}
        rm -f $PROCESSED_DIR/PROCESSED-KOKOfold${i}-${lang}.done
        rm -rf $CORPUS_DIR/PROCESSED-KOKOfold${i}-${lang}
        rm -f $CORPUS_DIR/train-KOKOfold${i}-${lang}.done
        rm -f $CORPUS_DIR/train-KOKOfold${i}-${lang}.tar.gz
        tar -xzvf $PATH_TO_PROCESSED/processed_fold${i}/PROCESSED-KOKOfold${i}-${lang}.tar.gz
    done
done