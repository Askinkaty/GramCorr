#!/usr/bin/env bash

tail -n +2 $1 | cut -f 1,4,5,6 | LC_NUMERIC=en_US.utf-8 sort -k1,1 -k3,3gr -k4gr | \
gawk -f ./_98_evalhack.awk

#tail -n +2 $1  | cut -f1,4 | awk '{vals[$1][$2]+=1}END{for (val in vals){if (vals[val][0]!="") {zeros=vals[val][0]} else {zeros=0};if (vals[val][1]!="") {ones=vals[val][1]} else {ones=0}; print val,zeros,ones,zeros+ones}}' | sort -k3g -k2g | less -N
