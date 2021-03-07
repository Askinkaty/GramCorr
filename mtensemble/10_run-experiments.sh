#!/usr/bin/env bash

set -e
set -o pipefail

source functions

### Run 10-fold CV on the 'largest' data set(s)
for INDIR in "${INPUT_EXPERIMENT}"/*-full/*
do
    INDIR_GUESSER_IDS=$(basename "$INDIR")
    INDIR_BASE=$(basename "$(dirname "$INDIR")")

    for INFILE in "${INDIR}"/*.arff
    do
        OUTDIR="${OUTPUT_EXPERIMENT}/${INDIR_BASE}/${INDIR_GUESSER_IDS}"
        OUTFILE="${OUTDIR}/$(basename "${INFILE}" .arff)"

        echo "[$(basename "$0")] ${INFILE} ---> ${OUTFILE}" >&2
        if [ "${INFILE}" -nt "${OUTFILE}.eval" ]
        then
            mkdir -p "${OUTDIR}"
            weka_rf "${INFILE}" "${OUTFILE}" \
            || { rm "${OUTFILE}.eval"; exit 1; }
        else
            echo "[$(basename "$0")] ${OUTFILE}.eval exists." >&2
        fi
    done
done

### Run 10-fold CV on the '-fold' data set(s)
for INDIR in "${INPUT_EXPERIMENT}"/*folds/*
do
    INDIR_GUESSER_IDS=$(basename "$INDIR")
    INDIR_BASE=$(basename "$(dirname "$INDIR")")

    FOLD_FILES=( "${INDIR}"/*.arff )

    for FOLD in $(seq 0 $(( ${#FOLD_FILES[@]} - 1 ))  )
    do
        echo "Fold: $FOLD"
        TST_IDX=FOLD
        TST_FILE=${FOLD_FILES[$TST_IDX]}

        TRN_IDX_BGN=$(( FOLD + 1 ))
        TRN_IDX_END=$(( FOLD + ${#FOLD_FILES[@]} - 1 ))
        TRN_IDX=( $(for IDX in $(seq ${TRN_IDX_BGN} ${TRN_IDX_END}); do echo $(( IDX % ${#FOLD_FILES[@]} )); done ) )
        TRN_FILES=( $(for IDX in ${TRN_IDX[@]}; do echo "${FOLD_FILES[$IDX]}"; done) )

        #echo $FOLD $TRN_IDX_BGN $TRN_IDX_END
        #echo ${TST_FILE}
        #echo "${TRN_FILES[@]}"

        OUTDIR="${OUTPUT_EXPERIMENT}/${INDIR_BASE}/${INDIR_GUESSER_IDS}"
        OUTFILE="${OUTDIR}/$(basename "${TST_FILE}" .arff)"

        echo "[$(basename "$0")] ${TST_FILE} ---> ${OUTFILE}" >&2
        if [ "${TST_FILE}" -nt "${OUTFILE}.eval" ]
        then
            mkdir -p "${OUTDIR}"

            arffs2arff "${TRN_FILES[@]}" > "${OUTFILE}_trn.arff" && \
            echo "Created training file:${OUTFILE}_trn.arff" \
            && weka_rf "${OUTFILE}_trn.arff" "${OUTFILE}" "${TST_FILE}" \
            && rm "${OUTFILE}_trn.arff" \
            || { rm "${OUTFILE}.eval"; exit 1; }
        else
            echo "[$(basename "$0")] ${OUTFILE}.eval exists." >&2
        fi

    done
done
