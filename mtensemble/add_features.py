#!/usr/bin/env python3
"""
Read a moses output csv file, calculate additional features (per error id) and
add the columns to the input. Calculations are done on the confidence scores of
each guesser and the following featurea are calculated:
- normalize
- StandardScaler
- MaxAbsScaled


# Data format (2020)

 --- column ---        --- index:type ---
 error_id(err_id)      # 1:str
 type                  # 2:str
 error_length          # 3:int:[1-
 suggestion            # 4:str
 is_correct(class)     # 5:int:{0,1}
 --- GUESSERS (2 columns per guesser: 1,0,-1 and confidence score)
 10_gram_is_suggested    10_gram_score     # cols  6, 7
 1_gram_is_suggested     1_gram_score      # cols  8, 9
 3_gram_is_suggested     3_gram_score      # ...  10,11
 5_gram_is_suggested     5_gram_score      # ...  12,13
 spellcheker_suggested   spellcheker_score # ...  14,15


# Note
ideally (if input files corespond to splits in the dataset), lines belonging to
one error_id should all be within one file.
"""
# old data (from 2018)
#
# errorId errorType original_length correction corr_1_incorr_0
# test-w1/valid-out/nbest_1_suggested_-1_other_0_nothing test-w1/valid-out/nbest
# test-w3/valid-out/nbest_1_suggested_-1_other_0_nothing test-w3/valid-out/nbest
# test-w5-car10/valid-out/nbest_1_suggested_-1_other_0_nothing test-w5-car10/valid-out/nbest
# test-w5-car10/valid-out/nbest-car_1_suggested_-1_other_0_nothing test-w5-car10/valid-out/nbest-car
#

import argparse
import sys

import pandas as pd
from sklearn import preprocessing

NUM_INFO_COLUMNS = 5    # Fix the number of 'info' columns


def init_argparse() -> argparse.ArgumentParser:
    """Parse arguments. """
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION]",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    return parser


parser = init_argparse()
args = parser.parse_args()

# read the data into a pandas DataFrame
data = pd.read_csv(sys.stdin, sep='\t', header=0)
# make sure (some) columns have predictable names
data.rename(columns={data.columns[0]: "err_id", data.columns[4]: "class"},
            inplace=True)
# try to infer the number of guessers in the input file and do a sanity check
# for the number of guessers: obviously, the number should be an int.
num_guessers = (len(data.columns) - 5) / 2
assert num_guessers == int(num_guessers)
num_guessers = int(num_guessers)
print(f"{num_guessers} guessers detected.", file=sys.stderr)

# spellchecker_score is inverse to the others: 0 is good and ]0,+inf[ is bad
# where as others are ]-inf,0[ bad and 0 is good. fix this here:
if "spellcheker_score" in data.columns:
    data["spellcheker_score"] = data["spellcheker_score"] * -1
    # print("Inverted 'spellchecker_score'.")

for err_id in data["err_id"].unique():
    # print(err_id, file=sys.stderr)
    for guess_id in range(num_guessers):

        scores = data[data["err_id"] == err_id].iloc[:, NUM_INFO_COLUMNS + 1 +
                                                     (guess_id * 2)]

        # print("scores: ", list(scores))
        # print(scores.min(), scores.max())

        # classes = data[data["err_id"] == err_id]["class"]
        # print("classes: ", list(classes))

        # copy the scores to a new column
        data.loc[(data["err_id"] == err_id), f"conf_norm_{guess_id}"] = scores
        # set the values of the new column to min(scores of err_id) for all
        # rows where the guesser did not suggest any correction
        # (i.e. instead of knowing nothing - because of an empty value - for
        # suggestions from *other* guessers, regard the suggestion as equal to
        # the least likely one from this guesser.
        #
        # SET:
        #  * where class: -1 (i did not propose this correction)
        #  * confval: my best suported correction
        # to
        #  * confval: min of this guesser's supported corrections for this error_id
        data.loc[
            (data["err_id"] == err_id) &
            (data.iloc[:, NUM_INFO_COLUMNS + (guess_id * 2)] == -1),
            f"conf_norm_{guess_id}"] = scores.min()
        # update scores with new values
        scores_updated = data[data["err_id"] == err_id][f"conf_norm_{guess_id}"]
        # print(list(scores_updated))

        scores_norm = preprocessing.normalize(scores_updated.to_frame(), norm='l2', axis=0)
        data.loc[(data["err_id"] == err_id), f"conf_norm_{guess_id}"] = scores_norm

        scores_std = preprocessing.StandardScaler().fit_transform(scores_updated.to_frame())
        data.loc[(data["err_id"] == err_id), f"std_{guess_id}"] = scores_std

        # scores_delta = 1 - scores_norm
        # data.loc[(data["err_id"] == err_id), f"delta_{guess_id}"] = scores_delta

        # scores_quant = preprocessing.QuantileTransformer(n_quantiles=10,
        #                                                  random_state=0).fit_transform(scores_updated.to_frame())
        # data.loc[(data["err_id"] == err_id), f"quant_{guess_id}"] = scores_quant

        scores_maxabs = preprocessing.MaxAbsScaler().fit_transform(scores_updated.to_frame())
        data.loc[(data["err_id"] == err_id), f"maxabs_{guess_id}"] = scores_maxabs

data.to_csv(sys.stdout, index=False)
