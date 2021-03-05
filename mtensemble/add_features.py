#!/usr/bin/env python3
"""
Read a moses output csv file, calculate additional features (per error id) and
add the columns to the input. Calculations are done on the confidence scores of
each guesser and the following features are calculated:
- normalize
- StandardScaler
- MaxAbsScaled
(scaling features is done by considering sets of lines belonging to identical
error_ids)


# Data format (2020)

 --- column ---        --- in_index:type:range ---
 *error_id(err_id)     # 1:str
 *type                 # 2:str
 error_length          # 3:int:[1-
 *suggestion           # 4:str
 is_correct(class)     # 5:int:{0,1}
 --- GUESSERS (2 columns per guesser: 1,0,-1 and confidence score)
 10_gram_is_suggested    10_gram_score     # cols  6, 7
 1_gram_is_suggested     1_gram_score      # cols  8, 9
 3_gram_is_suggested     3_gram_score      # ...  10,11
 5_gram_is_suggested     5_gram_score      # ...  12,13
 spellcheker_suggested   spellcheker_score # ...  14,15


# Note
- ideally (if input files corespond to splits in the dataset), lines belonging
  to one error_id should all be within one file.
- (*) for the output the columns err_id, type, and suggestion are sanitized.
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
import logging
import sys

import pandas as pd
import numpy as np
from sklearn import preprocessing

NUM_INFO_COLUMNS = 5    # Fix the number of 'info' columns


def init_argparse() -> argparse.ArgumentParser:
    """Parse arguments. """
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] < INPUT > OUTPUT",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--guesser_ids", required=False,
        help="Comma seperated list of guessers for which to output the \
        corresponding columns", metavar="0,2,..."
    )
    return parser


logging.basicConfig(
    stream=sys.stderr,
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(module)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)
parser = init_argparse()
args = parser.parse_args()

# Read the data into a pandas DataFrame
data = pd.read_csv(sys.stdin, sep='\t', header=0)

# Make sure (some) columns have predictable names
data.rename(columns={data.columns[0]: "err_id", data.columns[4]: "class"},
            inplace=True)

# Try to infer the number of guessers in the input file and do a sanity check
# for the number of guessers: obviously, the number should be an int.
num_guessers = (len(data.columns) - 5) / 2
assert num_guessers == int(num_guessers)
num_guessers = int(num_guessers)
log.debug(f"{num_guessers} guessers detected.")

# Set the guesser_ids to use when writing the output
if args.guesser_ids:
    guesser_ids = [int(gid.strip()) for gid in args.guesser_ids.split(',')]
else:
    guesser_ids = list(range(num_guessers))
assert max(guesser_ids) <= num_guessers

# spellchecker_score is inverse to the others: 0 is good and ]0,+inf[ is bad
# where as others are ]-inf,0[ bad and 0 is good. fix this here:
if "spellcheker_score" in data.columns:
    data["spellcheker_score"] = data["spellcheker_score"] * -1
    log.debug("Inverted 'spellchecker_score'.")

for err_id in data["err_id"].unique():
    # print(err_id, file=sys.stderr)
    for guess_id in range(num_guessers):

        scores = data[data["err_id"] == err_id].iloc[:, NUM_INFO_COLUMNS + 1 +
                                                     (guess_id * 2)]

        # print("scores: ", list(scores))
        # print(scores.min(), scores.max())

        # classes = data[data["err_id"] == err_id]["class"]
        # print("classes: ", list(classes))

        # Copy the scores to a new column
        data.loc[(data["err_id"] == err_id), f"conf_norm_{guess_id}"] = scores
        # Set the values of the new column to min(scores of err_id) for all
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

num_features = (len(data.columns) - NUM_INFO_COLUMNS - num_guessers*2) / num_guessers
assert num_features == int(num_features)
num_features = int(num_features)

range_info_columns = list(range(NUM_INFO_COLUMNS))
range_org_columns_suggested = np.array([NUM_INFO_COLUMNS+gid*2 for gid in guesser_ids])
range_org_columns_score = range_org_columns_suggested + 1
range_feats = np.array([NUM_INFO_COLUMNS+2*num_guessers+gid*num_features for gid in guesser_ids])
range_feats = [elem for fid in range(num_features) for elem in list(range_feats + fid)]

# print(range_info_columns + list(range_org_columns_suggested) +
#       list(range_org_columns_score) + range_feats)

guesser_columns = [data.columns[cid] for cid in sorted(list(range_info_columns) +
                         list(range_org_columns_suggested) +
                         list(range_org_columns_score) + list(range_feats))]

data.to_csv(sys.stdout, index=False, sep="\t", columns=guesser_columns)
