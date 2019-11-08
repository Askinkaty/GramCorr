#!/usr/bin/env bash

set -e

MODEL=1_gram
CORPUS=KOKO
CHAR=''

BASE_DIR=/hltsrv0/a.katinskaia/${CORPUS}
CUR_MODEL_DIR=$BASE_DIR/ALL_MODELS/$MODEL


PATH_TO_PROCESSED=$BASE_DIR/DATA/processed${CHAR}

CORPUS_DIR=/hltsrv2/SMT_TRAINING_SYSTEM/CORPUS_REPOSITORY
MODELS_DIR=/hltsrv2/SMT_TRAINING_SYSTEM/MODELS_REPOSITORY

TRAIN_DIR=$CUR_MODEL_DIR/TRAIN-${CORPUS}
TEST_DIR=$CUR_MODEL_DIR/TEST-${CORPUS}
TUNE_DIR=$CUR_MODEL_DIR/TUNE-${CORPUS}
MY_MODEL_DIR=$CUR_MODEL_DIR/MODELS

END=9
mkdir -p $CUR_MODEL_DIR
mkdir -p $TRAIN_DIR
mkdir -p $TEST_DIR
mkdir -p $TUNE_DIR
mkdir -p $MY_MODEL_DIR
DEV=($(seq $END -1 0))

for i in $(seq 0 $END)
do
    cd $PATH_TO_PROCESSED/processed_fold${i}
    tar -xzvf PROCESSED-${CORPUS}fold${i}-en.tar.gz
    tar -xzvf PROCESSED-${CORPUS}fold${i}-de.tar.gz
done

for i in $(seq 0 $END)
do
    echo 'fold: '${i}
    dev_ind=${DEV[${i}]}
    echo 'dev set: '${dev_ind}
    mkdir -p $TRAIN_DIR/fold${i}
    mkdir -p $TUNE_DIR/fold${i}
    mkdir -p $TEST_DIR/fold${i}


    cp $BASE_DIR/ActualParameters.cfg $TRAIN_DIR/fold${i}
    # echo "_MAX_PHRASE_LENGTH_=3" >> $TRAIN_DIR/fold${i}/ActualParameters.cfg # 3-gram
    # echo "_MAX_PHRASE_LENGTH_=5" >> $TRAIN_DIR/fold${i}/ActualParameters.cfg # 5-gram
    # echo "_MAX_PHRASE_LENGTH_=10" >> $TRAIN_DIR/fold${i}/ActualParameters.cfg # 10-gram
    echo "_MAX_PHRASE_LENGTH_=1" >> $TRAIN_DIR/fold${i}/ActualParameters.cfg # 1-gram
    cp $TRAIN_DIR/fold${i}/ActualParameters.cfg $TUNE_DIR/fold${i}
    cp $TRAIN_DIR/fold${i}/ActualParameters.cfg $TEST_DIR/fold${i}

    cp $PATH_TO_PROCESSED/processed_fold${i}/PROCESSED-${CORPUS}fold${i}-en/train.en $TEST_DIR/fold${i}
    cp $PATH_TO_PROCESSED/processed_fold${i}/PROCESSED-${CORPUS}fold${i}-de/train.de $TEST_DIR/fold${i}

    cp $PATH_TO_PROCESSED/processed_fold${dev_ind}/PROCESSED-${CORPUS}fold${dev_ind}-en/train.en $TUNE_DIR/fold${i}
    cp $PATH_TO_PROCESSED/processed_fold${dev_ind}/PROCESSED-${CORPUS}fold${dev_ind}-de/train.de $TUNE_DIR/fold${i}

    for j in $(seq 0 $END)
    do
        echo ${j}
        if [[ ${j} -ne ${i} && ${j} -ne ${dev_ind} ]]
        then
            echo 'train set: '${j}
            touch $TRAIN_DIR/fold${i}/train.en
            touch $TRAIN_DIR/fold${i}/train.de
            cat $PATH_TO_PROCESSED/processed_fold${j}/PROCESSED-${CORPUS}fold${j}-en/train.en >> $TRAIN_DIR/fold${i}/train.en
            cat $PATH_TO_PROCESSED/processed_fold${j}/PROCESSED-${CORPUS}fold${j}-de/train.de >> $TRAIN_DIR/fold${i}/train.de
        fi
    done
    cd $TRAIN_DIR/fold${i}
    tar -czvf train-${CORPUS}fold${i}-en-de.tar.gz train.en train.de ActualParameters.cfg
    cp train-${CORPUS}fold${i}-en-de.tar.gz $CORPUS_DIR
    cd $CORPUS_DIR
    touch train-${CORPUS}fold${i}-en-de.ready
done

cd $CORPUS_DIR
for i in $(seq 0 $END)
do
    echo ${i}
    while ! test -f "train-${CORPUS}fold${i}-en-de.done"
    do
        sleep 60
        echo "Still waiting"
    done
    mkdir -p $MY_MODEL_DIR/fold${i}/
    mv -f $MODELS_DIR/MODELS-${CORPUS}fold${i}-en-de.tar.gz $MY_MODEL_DIR/fold${i}
    rm -f $MODELS_DIR/MODELS-${CORPUS}fold${i}-en-de.done
    rm -f $CORPUS_DIR/train-${CORPUS}fold${i}-en-de.done
    rm -f $CORPUS_DIR/train-${CORPUS}fold${i}-en-de.tar.gz
    cd $MY_MODEL_DIR/fold${i}/
    tar -xzvf MODELS-${CORPUS}fold${i}-en-de.tar.gz
    cp $MY_MODEL_DIR/fold${i}/MODELS-${CORPUS}fold${i}-en-de/moses.ini $TUNE_DIR/fold${i}
    cd $CORPUS_DIR
done




