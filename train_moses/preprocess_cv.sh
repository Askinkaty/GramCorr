#!/usr/bin/env bash

set -e

HOME_DIR=/hltsrv0/a.katinskaia
CORPUS=KOKO
CHAR='_char'
#CHAR=''
#PATH_TO_DATASET=$HOME_DIR/${CORPUS}/DATA/cv${CHAR}

#needed to expriment with processing split sentences
#PATH_TO_DATASET=$HOME_DIR/${CORPUS}/DATA/cv/to_preproc${CHAR}
PATH_TO_DATASET=$HOME_DIR/${CORPUS}/DATA/to_preproc${CHAR}


#PATH_TO_PREPARED_DATA=$HOME_DIR/${CORPUS}/DATA/prepared${CHAR}
#PATH_TO_PROCESSED=$HOME_DIR/${CORPUS}/DATA/processed${CHAR}

#needed to expriment with processing split sentences
PATH_TO_PREPARED_DATA=$HOME_DIR/$CORPUS/DATA/split_prepared${CHAR}
PATH_TO_PROCESSED=$HOME_DIR/${CORPUS}/DATA/split_processed${CHAR}


TMP=$HOME_DIR/${CORPUS}/tmp

END=9

CORPUS_DIR=/hltsrv2/QT21_PROCESSING_SYSTEM/CORPUS_REPOSITORY
PROCESSED_DIR=/hltsrv2/QT21_PROCESSING_SYSTEM/PROCESSED_REPOSITORY

rm -rf $PATH_TO_PREPARED_DATA

for i in $(seq 0 $END)
do
    echo ${i}
    mkdir -p $PATH_TO_PREPARED_DATA/fold${i}
#    cp $PATH_TO_DATASET/fold_${i}_source.txt $PATH_TO_PREPARED_DATA/fold${i}/train.en
#    cp $PATH_TO_DATASET/fold_${i}_target.txt $PATH_TO_PREPARED_DATA/fold${i}/train.de

    cp $PATH_TO_DATASET/fold${i}_source.txt $PATH_TO_PREPARED_DATA/fold${i}/train.en
    cp $PATH_TO_DATASET/fold${i}_target.txt $PATH_TO_PREPARED_DATA/fold${i}/train.de

    for lang in en de
    do
        chmod 755 $PATH_TO_PREPARED_DATA/fold${i}/train.${lang}
        mkdir -p $TMP
        cp $PATH_TO_PREPARED_DATA/fold${i}/train.${lang} $TMP
        cp $HOME_DIR/${CORPUS}/ActualParameters.cfg $TMP
        cd $TMP
        tar -czvf train-${CORPUS}fold${i}-${lang}.tar.gz train.${lang} ActualParameters.cfg
        cp train-${CORPUS}fold${i}-${lang}.tar.gz $CORPUS_DIR
        cd $CORPUS_DIR
        touch train-${CORPUS}fold${i}-${lang}.ready
    done
done


cd $CORPUS_DIR
for i in $(seq 0 $END)
do
    for lang in en de
    do
        while ! test -f "train-${CORPUS}fold${i}-${lang}.done"
        do
            sleep 60
            echo "Still waiting"
        done
        rm -rf $TMP
        mkdir -p $PATH_TO_PROCESSED/processed_fold${i}
        mv $PROCESSED_DIR/PROCESSED-${CORPUS}fold${i}-${lang}.tar.gz $PATH_TO_PROCESSED/processed_fold${i}
        rm -rf $PROCESSED_DIR/PROCESSED-${CORPUS}fold${i}-${lang}.done
        rm -rf $CORPUS_DIR/PROCESSED-${CORPUS}fold${i}-${lang}
        rm -rf $CORPUS_DIR/train-${CORPUS}fold${i}-${lang}.done
        rm -rf $CORPUS_DIR/train-${CORPUS}fold${i}-${lang}.tar.gz
        tar -xzvf $PATH_TO_PROCESSED/processed_fold${i}/PROCESSED-${CORPUS}fold${i}-${lang}.tar.gz
        cd $CORPUS_DIR
    done
done



for i in $(seq 0 $END)
do
    cd $PATH_TO_PROCESSED/processed_fold${i}
    tar -xzvf PROCESSED-${CORPUS}fold${i}-en.tar.gz
    tar -xzvf PROCESSED-${CORPUS}fold${i}-de.tar.gz
    mkdir -p $PATH_TO_PROCESSED/folds/fold${i}
    cp $PATH_TO_PROCESSED/processed_fold${i}/PROCESSED-${CORPUS}fold${i}-en/train.en $PATH_TO_PROCESSED/folds/fold${i}
    cp $PATH_TO_PROCESSED/processed_fold${i}/PROCESSED-${CORPUS}fold${i}-de/train.de $PATH_TO_PROCESSED/folds/fold${i}


done