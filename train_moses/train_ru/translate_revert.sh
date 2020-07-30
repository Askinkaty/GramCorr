#!/usr/bin/env bash

set -e


MODEL=10_gram
CORPUS=RUS
CHAR='_char'
#CHAR=''

BASE_DIR=/hltsrv0/a.katinskaia/${CORPUS}
CUR_MODEL_DIR=$BASE_DIR/ALL_MODELS_REVERSE/$MODEL


CORPUS_DIR=/hltsrv2/SMT_TRANSLATION_SYSTEM/CORPUS_REPOSITORY
TRANSLATION_DIR=/hltsrv2/SMT_TRANSLATION_SYSTEM/TRANSLATION_REPOSITORY


MY_MODEL_DIR=$CUR_MODEL_DIR/MODELS

#TEST_DIR=$CUR_MODEL_DIR/TEST-${CORPUS}
TEST_DIR=$CUR_MODEL_DIR/TEST_XML

TUNED_MODELS=$CUR_MODEL_DIR/TUNED_MODELS
#TRANSLATED=$CUR_MODEL_DIR/TRANSLATED-${CORPUS}
TRANSLATED=$CUR_MODEL_DIR/TRANSLATED-XML

TMP=$MY_MODEL_DIR/TMP


mkdir -p $TRANSLATED


cd $TUNED_MODELS
mkdir -p $TMP

cp $TEST_DIR/train.ru $TMP
cp $TEST_DIR/ActualParameters.cfg $TMP
cp $TUNED_MODELS/TUNING-${CORPUS}TR-ru-en/moses.ini $TMP
cd $TMP


#echo "_MAX_PHRASE_LENGTH_=3" >> ActualParameters.cfg # 3-gram
#echo "_MAX_PHRASE_LENGTH_=5" >> ActualParameters.cfg  # 5-gram
#echo "_MAX_PHRASE_LENGTH_=10" >> ActualParameters.cfg # 10-gram
#echo "_MAX_PHRASE_LENGTH_=1" >> ActualParameters.cfg # 1-gram

echo "_MOSES-PARAMETER_=-xml-input exclusive" >> ActualParameters.cfg # add parameter for xml markup

tar -czvf train-${CORPUS}-ru-en.tar.gz train.ru ActualParameters.cfg moses.ini
cp train-${CORPUS}-ru-en.tar.gz $CORPUS_DIR
rm -rf $TMP
cd $CORPUS_DIR
touch train-${CORPUS}-ru-en.ready

cd $CORPUS_DIR



while ! test -f "$CORPUS_DIR/train-${CORPUS}-ru-en.done"
do
    sleep 60
    echo "Still waiting"
done


mv $TRANSLATION_DIR/TRANSLATION-${CORPUS}-ru-en.tar.gz $TRANSLATED
rm -f $TRANSLATION_DIR/TRANSLATION-${CORPUS}-ru-en.done
rm -f $CORPUS_DIR/train-${CORPUS}-ru-en.tar.gz
rm -f $CORPUS_DIR/train-${CORPUS}-ru-en.done
cd $TRANSLATED
tar -xzvf TRANSLATION-${CORPUS}-ru-en.tar.gz
cd TRANSLATION-${CORPUS}-ru-en
mv * ../
cd ..
rm -rf TRANSLATION-${CORPUS}-ru-en
rm -r TRANSLATION-${CORPUS}-ru-en.tar.gz
cd $CORPUS_DIR