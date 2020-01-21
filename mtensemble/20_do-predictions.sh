set -e
. 99_utils

# Predict data

for MODELOPT in *-modelopt.xml
do
    OUT_BASE=$(basename $MODELOPT -modelopt.xml)
    DATA=$(echo ${OUT_BASE}.arff | sed 's/\(.*\)-.*/\1/').arff

    OUTFILE=${OUT_BASE}.preds
    [ "$MODELOPT" -nt "$OUTFILE" ] && \
        echo "Creating $OUTFILE" && \
        ${JAVA_CMD} $(xmllint --xpath 'string(/object[@name = "__root__"]/@class)' $MODELOPT) \
        -l ${MODELOPT} \
        -t ${DATA} \
        -T ${DATA} -classifications weka.classifiers.evaluation.output.prediction.CSV \
        > $OUTFILE
done

exit 0
