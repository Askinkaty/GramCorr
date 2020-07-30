#!/usr/bin/env bash
set -e

CORPUS=RUS
CHAR='_char'
#CHAR=''


#/hltsrv0/a.katinskaia/RUS/ALL_MODELS/1_gram/MODELS/MODELS-RUSTR-ru-en
declare -a arr=("train" "test" "dev")

set -e

MODEL=10_gram

BASE_DIR=/hltsrv0/a.katinskaia/${CORPUS}
CUR_MODEL_DIR=$BASE_DIR/ALL_MODELS_REVERSE/$MODEL


#PATH_TO_PROCESSED=$BASE_DIR/DATA/processed_tr${CHAR}
MY_MODEL_DIR=$CUR_MODEL_DIR/MODELS

TUNE_DIR=$CUR_MODEL_DIR/TUNE-${CORPUS}
CORPUS_DIR=/hltsrv2/SMT_TUNING_SYSTEM/CORPUS_REPOSITORY
TUNING_DIR=/hltsrv2/SMT_TUNING_SYSTEM/TUNING_REPOSITORY

TUNED_MODELS=$CUR_MODEL_DIR/TUNED_MODELS
TMP=$MY_MODEL_DIR/TMP
#
#mkdir -p $TUNED_MODELS
#chmod -R 755 $MY_MODEL_DIR
#
#mkdir -p $TMP
#cp $TUNE_DIR/train.en $TMP
#cp $TUNE_DIR/train.ru $TMP
#cp $TUNE_DIR/ActualParameters.cfg $TMP
#cp $TUNE_DIR/moses.ini $TMP
#chmod -R 755 $TMP
#cd $TMP
#sed -i "s|MODELS-${CORPUS}TR-ru-en|$MY_MODEL_DIR/MODELS-${CORPUS}TR-ru-en|" moses.ini
#tar -czvf train-${CORPUS}TR-ru-en.tar.gz train.en train.ru ActualParameters.cfg moses.ini
#cp train-${CORPUS}TR-ru-en.tar.gz $CORPUS_DIR
#cd $CORPUS_DIR
#touch train-${CORPUS}TR-ru-en.ready
#rm -rf $TMP
#
#
#while ! test -f "train-${CORPUS}TR-ru-en.done"
#do
#    sleep 60
#    echo "Still waiting"
#done
cd $TUNING_DIR
mv $TUNING_DIR/TUNING-${CORPUS}TR-ru-en.tar.gz $TUNED_MODELS
rm -rf $CORPUS_DIR/train-${CORPUS}TR-ru-en.done
rm -rf $CORPUS_DIR/train-${CORPUS}TR-ru-en.tar.gz
rm -rf $TUNING_DIR/TUNING-${CORPUS}TR-ru-en.done
cd $TUNED_MODELS
tar -xzvf TUNING-${CORPUS}TR-ru-en.tar.gz
cd $CORPUS_DIR
