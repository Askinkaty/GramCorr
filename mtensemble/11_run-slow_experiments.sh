set -e
. 99_utils

# Run 3-fold CV (with some slow algorithms) on the data.

for DATA in nn*.arff
do
    OUT_BASE=$(basename $DATA .arff)

    OUTFILE=${OUT_BASE}-mlp.cvres
    OUTFILE_MODEL=$(basename $OUTFILE .cvres)-modelopts.xml
    [ "$DATA" -nt "$OUTFILE" ] && \
        echo "Creating $OUTFILE" && \
        ${JAVA_CMD} weka.classifiers.functions.MultilayerPerceptron \
        -L 0.3 -M 0.2 -N 500 -V 25 -S 1 -E 20 -H t -t ${DATA} -s 1 -x 3 -v -o -batch-size 10 \
        -d ${OUTFILE_MODEL} \
        > $OUTFILE

    OUTFILE=${OUT_BASE}-mlp_weighted.cvres
    OUTFILE_MODEL=$(basename $OUTFILE .cvres)-modelopts.xml
    [ "$DATA" -nt "$OUTFILE" ] && \
        echo "Creating $OUTFILE" && \
        ${JAVA_CMD} weka.classifiers.meta.WeightedInstancesHandlerWrapper \
        -S 1 -t ${DATA} -s 1 -x 3 -v -o -W weka.classifiers.functions.MultilayerPerceptron -- \
        -L 0.3 -M 0.2 -N 500 -V 25 -S 1 -E 20 -H t -batch-size 10 \
        -d ${OUTFILE_MODEL} \
        > $OUTFILE

    OUTFILE=${OUT_BASE}-smo.cvres
    OUTFILE_MODEL=$(basename $OUTFILE .cvres)-modelopts.xml
    [ "$DATA" -nt "$OUTFILE" ] && \
        echo "Creating $OUTFILE" && \
        ${JAVA_CMD} weka.classifiers.functions.SMO -C 1.0 -L 0.001 -P 1.0E-12 -N 1 -V -1 -W 1 \
        -K "weka.classifiers.functions.supportVector.RBFKernel -C 250007 -G 0.01" \
        -calibrator "weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4" \
        -t ${DATA} -s 1 -x 3 -v -o \
        -d ${OUTFILE_MODEL} \
        > $OUTFILE
done

exit 0
