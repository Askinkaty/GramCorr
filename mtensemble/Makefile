SHELL:=/bin/bash

DATA_CSV_DIRECT_SOURCES = nn-1.csv nn-2.csv nn-3.csv
DATA_CSV_01_SOURCES = nn-1_01.csv nn-2_01.csv nn-3_01.csv
DATA_CSV_SOURCES = $(DATA_CSV_DIRECT_SOURCES) $(DATA_CSV_01_SOURCES)
DATA_ARFF_TARGETS = $(patsubst %.csv,%.arff,$(DATA_CSV_SOURCES))


.PHONY : all 
all : data

$(DATA_CSV_DIRECT_SOURCES) : _00_getnprep-data.sh
	./_00_getnprep-data.sh $@
	@touch $@

nn-%_01.csv : nn-%.csv 01_adddata.sh _01_adddata.py
	./01_adddata.sh

%.arff : %.csv _09_mkarff.sh
	./_09_mkarff.sh $<

$(DATA_ARFF_TARGETS) : $(DATA_CSV_SOURCES)

data : $(DATA_ARFF_TARGETS) 

.PHONY : experiments
experiments : data 10_run-experiments.sh
	./10_run-experiments.sh

.PHONY: experiments-clean
experiments-clean :
	rm -f *.cvres
