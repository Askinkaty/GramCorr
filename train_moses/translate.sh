#!/usr/bin/env bash

set -e


MODEL=10_gram
CORPUS=KOKO
CHAR='_char'


BASE_DIR=/hltsrv0/a.katinskaia/${CORPUS}
#CUR_MODEL_DIR=$BASE_DIR/ALL_MODELS/$MODEL
CUR_MODEL_DIR=$BASE_DIR/ALL_REVERTED_MODELS/$MODEL


CORPUS_DIR=/hltsrv2/SMT_TRANSLATION_SYSTEM/CORPUS_REPOSITORY
TRANSLATION_DIR=/hltsrv2/SMT_TRANSLATION_SYSTEM/TRANSLATION_REPOSITORY


MY_MODEL_DIR=$CUR_MODEL_DIR/MODELS

TEST_DIR=$CUR_MODEL_DIR/TEST-${CORPUS}
#XML_DIR=$BASE_DIR/ALL_MODELS/xml_test
XML_DIR=$BASE_DIR/ALL_MODELS/char_test

TUNED_MODELS=$CUR_MODEL_DIR/TUNED_MODELS
TRANSLATED=$CUR_MODEL_DIR/TRANSLATED-XML-${CORPUS}
#TRANSLATED=$CUR_MODEL_DIR/TRANSLATED-${CORPUS}
TMP=$MY_MODEL_DIR/TMP


END=9
mkdir -p $TRANSLATED

for i in $(seq 0 $END)
do
    cd $TUNED_MODELS/fold${i}
#    mv TUNING-${CORPUS}5fold${i}-de-en.tar.gz TUNING-${CORPUS}fold${i}-de-en.tar.gz
#    tar -xzvf TUNING-${CORPUS}fold${i}-de-en.tar.gz
    mkdir -p $TMP

#    cp $TEST_DIR/fold${i}/train.en $TMP
    cp $XML_DIR/fold${i}/train.de $TMP # take the file with xml markup for decoding

    cp $TEST_DIR/fold${i}/ActualParameters.cfg $TMP
    cp $TUNED_MODELS/fold${i}/TUNING-${CORPUS}10fold${i}-de-en/moses.ini $TMP
    cd $TMP

    echo "_MOSES-PARAMETER_=-xml-input exclusive" >> ActualParameters.cfg # add parameter for xml markup

    tar -czvf train-${CORPUS}XMLfold${i}-de-en.tar.gz train.de ActualParameters.cfg moses.ini
    cp train-${CORPUS}XMLfold${i}-de-en.tar.gz $CORPUS_DIR
    rm -rf $TMP
    cd $CORPUS_DIR
    touch train-${CORPUS}XMLfold${i}-de-en.ready
done

cd $CORPUS_DIR
for i in $(seq 0 $END)
do
    while ! test -f "$CORPUS_DIR/train-${CORPUS}XMLfold${i}-de-en.done"
    do
        sleep 60
        echo "Still waiting"
    done
    mkdir -p $TRANSLATED/fold${i}
    mv $TRANSLATION_DIR/TRANSLATION-${CORPUS}XMLfold${i}-de-en.tar.gz $TRANSLATED/fold${i}
    rm -f $TRANSLATION_DIR/TRANSLATION-${CORPUS}XMLfold${i}-de-en.done
    rm -f $CORPUS_DIR/train-${CORPUS}XMLfold${i}-de-en.tar.gz
    rm -f $CORPUS_DIR/train-${CORPUS}XMLfold${i}-de-en.done
    cd $TRANSLATED/fold${i}
    tar -xzvf TRANSLATION-${CORPUS}XMLfold${i}-de-en.tar.gz
    cd TRANSLATION-${CORPUS}XMLfold${i}-de-en
    mv * ../
    cd ..
    rm -rf TRANSLATION-${CORPUS}XMLfold${i}-de-en
    rm -r TRANSLATION-${CORPUS}XMLfold${i}-de-en.tar.gz
    cd $CORPUS_DIR
done