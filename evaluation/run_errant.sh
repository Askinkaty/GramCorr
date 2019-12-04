#!/usr/bin/env bash

ERRANT_DIR=$HOME/GramCorr/boyd-lowresource/errant
#CORPUS=KOKO
CORPUS=Koko_xml
DATA=cv
MODEL=1_gram
folds=9

#TRANS_DIR=$HOME/GramCorr/translated/${CORPUS}/${MODEL}/translated
TRANS_DIR=$HOME/GramCorr/translated/${CORPUS}/${MODEL}
DATA_DIR=$HOME/GramCorr/translate/Koko/word_test

#DATA_DIR=$HOME/GramCorr/translated/RUS/Revita


prec=0
recall=0
f=0

for i in $(seq 0 $folds)
do
#     create m2 format for reference
    mkdir -p $ERRANT_DIR/TMP_XML
    cd $ERRANT_DIR
#    python3 ./parallel_to_m2.py -orig $DATA_DIR/source.ru -cor $DATA_DIR/target.ru -out ref_ru_m2 -lang ru
    python3 ./parallel_to_m2.py -orig $DATA_DIR/fold${i}/train.en -cor $DATA_DIR/fold${i}/train.de -out ref_fold1${i}_m2 -lang de

    mv ref_fold${i}_m2 $ERRANT_DIR/TMP_XML/$MODEL/
#
#    # create m2 format for hypothesis
#    python3 ./parallel_to_m2.py -orig $DATA_DIR/fold${i}/train.en -cor $TRANS_DIR/fold${i}/train.en.trans.de -out hp_fold${i}_m2 -lang de
#    mv hp_fold${i}_m2 $ERRANT_DIR/TMP/$MODEL
#     python3 ./compare_m2.py -hyp $ERRANT_DIR/TMP/$MODEL/hp_fold${i}_m2 -ref $ERRANT_DIR/TMP/$MODEL/ref_fold${i}_m2 #>> out.csv


done