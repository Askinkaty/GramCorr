#!/usr/bin/env bash
# 
# bash dependencies:
# - [ FILE1 -nt FILE2 ]
# - string substitution
# - arrays

set -e
set -o pipefail

source functions

## Add features to data (cf. add_features.py for details).
for INDIR in "${INPUT_PREPROC}"/*-full "${INPUT_PREPROC}"/*folds
do
    INDIR_BASE=$(basename "$INDIR")

    for INFILE in "${INDIR}"/*.csv
    do
        OUTDIR="${INPUT_FEATURE}/${INDIR_BASE}"
        OUTFILE="${OUTDIR}/$(basename "${INFILE}" .csv).csv"

        echo "[$(basename "$0")] ${INFILE} ---> ${OUTFILE}" >&2
        if [ "${INFILE}" -nt "${OUTFILE}" ]
        then
            mkdir -p "${OUTDIR}"
            ./add_features.py < "${INFILE}" > "${OUTFILE}" \
            || { rm "${OUTFILE}"; exit 1; }
        else
            echo "[$(basename "$0")] ${OUTFILE} exists." >&2
        fi
    done
done

## Select guessers, i.e. column/row-combinations from the data (cf. select_guessers.py for details).
for INDIR in "${INPUT_FEATURE}"/*
do
    INDIR_BASE=$(basename "$INDIR")

    for INFILE in "${INDIR}"/*.csv
    do

        for GUESSER_IDS in 0 1 2 3 4 0_4 2_3 ALL
        do
            case $GUESSER_IDS in
                ALL)
                    ADD_FEATURES_ARGS=()
                    SELECT_GUESSERS_ARGS=()
                    CSVLOADER_EXTRA_ARGS=()
                    ;;
                ?)
                    ADD_FEATURES_ARGS=(--guesser_ids "${GUESSER_IDS//_/,}")
                    SELECT_GUESSERS_ARGS=("${ADD_FEATURES_ARGS[@]}")
                    CSVLOADER_EXTRA_ARGS=( \
                        -N "6" \
                        -L "6:-1,0,1" \
                        -R "7" \
                    )
                    ;;
                ?_?)
                    ADD_FEATURES_ARGS=(--guesser_ids "${GUESSER_IDS//_/,}")
                    SELECT_GUESSERS_ARGS=("${ADD_FEATURES_ARGS[@]}")
                    CSVLOADER_EXTRA_ARGS=( \
                        -N "6,8" \
                        -L "6,8:-1,0,1" \
                        -R "7,9" \
                    )
                    ;;
                ?_?_?)
                    ADD_FEATURES_ARGS=(--guesser_ids "${GUESSER_IDS//_/,}")
                    SELECT_GUESSERS_ARGS=("${ADD_FEATURES_ARGS[@]}")
                    CSVLOADER_EXTRA_ARGS=( \
                        -N "6,8,10" \
                        -L "6,8,10:-1,0,1" \
                        -R "7,9,11" \
                    )
                    ;;
                ?_?_?_?)
                    ADD_FEATURES_ARGS=(--guesser_ids "${GUESSER_IDS//_/,}")
                    SELECT_GUESSERS_ARGS=("${ADD_FEATURES_ARGS[@]}")
                    CSVLOADER_EXTRA_ARGS=( \
                        -N "6,8,10,12" \
                        -L "6,8,10,12:-1,0,1" \
                        -R "7,9,11,13" \
                    )
                    ;;
            esac

            # Convert .csv feature set file to .arff input file.
            OUTDIR="${INPUT_EXPERIMENT}/${INDIR_BASE}/${GUESSER_IDS}"
            OUTFILE="${OUTDIR}/$(basename "${INFILE}" .csv).arff"

            echo "[$(basename "$0")] ${INFILE} ---> ${OUTFILE}" >&2
            if [ "${INFILE}" -nt "${OUTFILE}" ]
            then
                mkdir -p "${OUTDIR}"
                ./select_guessers.py "${SELECT_GUESSERS_ARGS[@]}" < "${INFILE}" > "${OUTFILE}.csv" \
                && csv2arff "${OUTFILE}.csv" "${OUTFILE}" "${CSVLOADER_EXTRA_ARGS[@]}" \
                && rm "${OUTFILE}.csv" \
                || { rm "${OUTFILE}"; exit 1; }
            else
                echo "[$(basename "$0")] ${OUTFILE} exists." >&2
            fi
        done
    done
done
#
###
