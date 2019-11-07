#!/usr/bin/env bash

set -e

KOKO_DIR=/hltsrv0/a.katinskaia/KOKO
PATH_TO_PROCESSED=$KOKO_DIR/processed

CORPUS_DIR=/hltsrv2/SMT_TRAINING_SYSTEM/CORPUS_REPOSITORY
MODELS_DIR=/hltsrv2/SMT_TRAINING_SYSTEM/MODELS_REPOSITORY

DATA_DIR=$KOKO_DIR/processed/processed_train
TRAIN_DIR=$KOKO_DIR/TRAIN-KOKO
TEST_DIR=$KOKO_DIR/TEST-KOKO
TUNE_DIR=$KOKO_DIR/TUNE-KOKO
MY_MODEL_DIR=$KOKO_DIR/MODELS

END=9
mkdir -p $TRAIN_DIR
mkdir -p $TEST_DIR
mkdir -p $TUNE_DIR
mkdir -p $MY_MODEL_DIR
DEV=($(seq $END -1 0))

for i in $(seq 0 $END)
do
    cd $PATH_TO_PROCESSED/processed_fold${i}
    tar -xzvf PROCESSED-KOKOfold${i}-en.tar.gz
    tar -xzvf PROCESSED-KOKOfold${i}-de.tar.gz
done

for i in $(seq 0 $END)
do
    echo 'fold: '${i}
    dev_ind=${DEV[${i}]}
    echo 'dev set: '${dev_ind}
    mkdir -p $TRAIN_DIR/fold${i}
    mkdir -p $TUNE_DIR/fold${i}
    mkdir -p $TEST_DIR/fold${i}


    cp $KOKO_DIR/ActualParameters.cfg $TRAIN_DIR/fold${i}
    # echo "_MAX_PHRASE_LENGTH_=3" >> $TRAIN_DIR/fold${i}/ActualParameters.cfg
    # echo "_MAX_PHRASE_LENGTH_=5" >> $TRAIN_DIR/fold${i}/ActualParameters.cfg
    # echo "_MAX_PHRASE_LENGTH_=10" >> $TRAIN_DIR/fold${i}/ActualParameters.cfg
    echo "_MAX_PHRASE_LENGTH_=1" >> $TRAIN_DIR/fold${i}/ActualParameters.cfg
    cp $TRAIN_DIR/fold${i}/ActualParameters.cfg $TUNE_DIR/fold${i}
    cp $TRAIN_DIR/fold${i}/ActualParameters.cfg $TEST_DIR/fold${i}

    cp $PATH_TO_PROCESSED/processed_fold${i}/PROCESSED-KOKOfold${i}-en/train.en $TEST_DIR/fold${i}
    cp $PATH_TO_PROCESSED/processed_fold${i}/PROCESSED-KOKOfold${i}-de/train.de $TEST_DIR/fold${i}

    cp $PATH_TO_PROCESSED/processed_fold${dev_ind}/PROCESSED-KOKOfold${dev_ind}-en/train.en $TUNE_DIR/fold${i}
    cp $PATH_TO_PROCESSED/processed_fold${dev_ind}/PROCESSED-KOKOfold${dev_ind}-de/train.de $TUNE_DIR/fold${i}

    for j in $(seq 0 $END)
    do
        echo ${j}
        if [[ ${j} -ne ${i} && ${j} -ne ${dev_ind} ]]
        then
            echo 'train set: '${j}
            touch $TRAIN_DIR/fold${i}/train.en
            touch $TRAIN_DIR/fold${i}/train.de
            cat $PATH_TO_PROCESSED/processed_fold${j}/PROCESSED-KOKOfold${j}-en/train.en >> $TRAIN_DIR/fold${i}/train.en
            cat $PATH_TO_PROCESSED/processed_fold${j}/PROCESSED-KOKOfold${j}-de/train.de >> $TRAIN_DIR/fold${i}/train.de
        fi
    done
    cd $TRAIN_DIR/fold${i}
    tar -czvf train-KOKOfold${i}-en-de.tar.gz train.en train.de ActualParameters.cfg
    cp train-KOKOfold${i}-en-de.tar.gz $CORPUS_DIR
    cd $CORPUS_DIR
    touch train-KOKOfold${i}-en-de.ready
done

cd $CORPUS_DIR
for i in $(seq 0 $END)
do
    echo ${i}
    while ! test -f "train-KOKOfold${i}-en-de.done"
    do
        sleep 60
        echo "Still waiting"
    done
    mkdir -p $MY_MODEL_DIR/fold${i}/
    mv -f $MODELS_DIR/MODELS-KOKOfold${i}-en-de.tar.gz $MY_MODEL_DIR/fold${i}
    rm -f $MODELS_DIR/MODELS-KOKOfold${i}-en-de.done
    rm -f $CORPUS_DIR/train-KOKOfold${i}-en-de.done
    rm -f $CORPUS_DIR/train-KOKOfold${i}-en-de.tar.gz
    cd $MY_MODEL_DIR/fold${i}/
    tar -xzvf MODELS-KOKOfold${i}-en-de.tar.gz
    cp $MY_MODEL_DIR/fold${i}/MODELS-KOKOfold${i}-en-de/moses.ini $TUNE_DIR/fold${i}
    cd $CORPUS_DIR
done




