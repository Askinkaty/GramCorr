#!/usr/bin/env bash
# 
# bash dependencies:
# - [ FILE1 -nt FILE2 ]
# - string substitution

set -e

. functions
. local

mkdir -p "${INPUT_FEATSET}"


## Add features to data (cf. add_features.py for details).

for INFILE in "${INPUT_READY}"/*.csv
do
    echo ${INFILE}

    OUTFILE="${INPUT_FEATSET}/$(basename "$INFILE" .csv).csv"
    if [ "${INFILE}" -nt "${OUTFILE}" ]
    then
        echo "[$(basename "$0")] Computing ${OUTFILE}"
        ./add_features.py \
        < "${INFILE}" > "${OUTFILE}"
    else
        echo "[$(basename "$0")] Doing nothing for ${INFILE}." >&2
    fi
done


# Convert all .csv feature set files to .arff input files.

for INFILE in "${INPUT_FEATSET}"/*.csv
do
    OUTFILE="${INPUT_FEATSET}"/$(basename "${INFILE}" .csv).arff
    convert01 "${INFILE}" "${OUTFILE}" \
    || ( rm "${OUTFILE}"; exit 1 )
done
#
###
