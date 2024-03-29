#!/usr/bin/env python3
"""
Read a moses output csv file, calculate additional features (per error id) and
add the columns to the input. Calculations are done on the confidence scores of
each guesser individually and the following features are calculated:
- normalize
- StandardScaler
- MaxAbsScaled
(scaling features is done by considering sets of lines that belong to identical
error_ids and where the guesser suggested a correction)


# Data format (2020)

 --- column ---        --- in_index:type:range ---
 *error_id(err_id)     # 1:str
 *type                 # 2:str
 error_length          # 3:int:[1-
 *suggestion           # 4:str
 is_correct(class)     # 5:int:{0,1}
 --- GUESSERS (2 columns per guesser: 1,0,-1 and confidence score)
 10_gram_is_suggested   10_gram_score      10_gram_rank # cols  6, 7, 8
  1_gram_is_suggested    1_gram_score       1_gram_rank # cols  9,10,11
  3_gram_is_suggested    3_gram_score       3_gram_rank # ...  12,13,14
  5_gram_is_suggested    5_gram_score       5_gram_rank # ...  15,16,17
 spellcheker_suggested  spellcheker_score  spellchecker_rank # 18,19,20

The values in the *_is_suggestd columns mean:
*  1: this is the best suggested correction from this system
* -1: this correction was suggested by the system but not as the best one
*  0: the system had NO suggestion for this error_id (and the score is also 0)


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
NUM_COLS_PER_GUESSER = 3
UNKNOWN_VALUE = "?"     # Set the character to represent missing/unknown values


def init_argparse() -> argparse.ArgumentParser:
    """Parse arguments. """
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] < INPUT > OUTPUT",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        help="Increase verbosity")
    parser.add_argument(
        "--guesser_ids", required=False,
        help="Comma seperated list of guessers for which to output the \
        corresponding columns", metavar="0,2,..."
    )
    return parser


parser = init_argparse()
args = parser.parse_args()
if not args.verbose:
    log_level = logging.INFO
else:
    log_level = logging.DEBUG

logging.basicConfig(
    stream=sys.stderr,
    level=log_level,
    format='%(asctime)s %(levelname)s %(module)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)

# Read the data into a pandas DataFrame
data = pd.read_csv(sys.stdin, sep='\t', header=0)

# Make sure (some) columns have predictable names
data.rename(columns={data.columns[0]: "err_id", data.columns[4]: "class"},
            inplace=True)

# Sanitize the err_id column (remove space chars)
data.err_id.replace(to_replace=" ", value="_", regex=True, inplace=True)
data.type.replace(to_replace=r"[\ \"']", value="_", regex=True, inplace=True)
data.suggestion.replace(to_replace=r"[\ \"']", value="_", regex=True, inplace=True)
log.info(f"Sanitized err_id,type,suggestion column(s).")

# Try to infer the number of guessers in the input file and do a sanity check
# for the number of guessers: obviously, the number should be an int.
num_guessers = (len(data.columns) - NUM_INFO_COLUMNS) / NUM_COLS_PER_GUESSER
assert num_guessers == int(num_guessers)
num_guessers = int(num_guessers)
log.info(f"{num_guessers} guessers detected.")

# Set the guesser_ids to use when writing the output
if args.guesser_ids:
    guesser_ids = [int(gid.strip()) for gid in args.guesser_ids.split(',')]
else:
    guesser_ids = list(range(num_guessers))
assert max(guesser_ids) <= num_guessers

err_ids = list(data["err_id"].unique())
for err_num, err_id in enumerate(err_ids):
    log.debug(f"{err_id}")
    if err_num % 100 == 0:
        log.info(f"{err_num} of {len(err_ids)}.")

    for guess_id in range(num_guessers):

        # Construct a few vectorized selectors to speed up look-ups
        err_id_selector = data.err_id.isin([err_id])
        # class_minus_1_selector = data[
        #     data.columns[NUM_INFO_COLUMNS + (guess_id * 2)]].isin([-1])
        class_0_selector = data[
            data.columns[NUM_INFO_COLUMNS + (guess_id * NUM_COLS_PER_GUESSER)]].isin([0])
        scores_selector = err_id_selector & ~class_0_selector
        scores_selector_sum = scores_selector.abs().sum()
        scores_unknown_selector = err_id_selector & class_0_selector

        scores = data[scores_selector][
                          data.columns[NUM_INFO_COLUMNS + 1 + (guess_id *
                                                               NUM_COLS_PER_GUESSER)]]
        scores_unknown = scores_unknown_selector.sum() * [UNKNOWN_VALUE]
        log.debug("scores: %s", list(scores))
        log.debug("scores: min/max: %s/%s ", scores.min(), scores.max())

        # classes = data[scores_selector]["class"]
        # print("classes: ", list(classes))

        ranks = data[scores_selector][
                          data.columns[NUM_INFO_COLUMNS + 2 + (guess_id *
                                                               NUM_COLS_PER_GUESSER)]]

        column = f"score_norm_{guess_id}"
        data.loc[scores_unknown_selector, column] = scores_unknown
        if scores_selector_sum > 0:
            scores_feat = preprocessing.normalize(scores.to_frame(), norm='l2', axis=0)
            # scores_norm = scores_feat
            data.loc[scores_selector, column] = scores_feat

        column = f"score_std_{guess_id}"
        data.loc[scores_unknown_selector, column] = scores_unknown
        if scores_selector_sum > 0:
            scores_feat = preprocessing.StandardScaler().fit_transform(scores.to_frame())
            data.loc[scores_selector, column] = scores_feat

        # Note: Make sure the scores are all aligned along the same axis/range.
        # column = f"score_delta_{guess_id}"
        # data.loc[scores_unknown_selector, column] = scores_unknown
        # if scores_selector_sum > 0:
        #     scores_feat = 1 - scores_norm
        #     data.loc[scores_selector, column] = scores_feat

        # column = f"score_quant_{guess_id}"
        # data.loc[scores_unknown_selector, column] = scores_unknown
        # if scores_selector_sum > 0:
        #     scores_feat = preprocessing.QuantileTransformer(
        #         n_quantiles=10, random_state=0).fit_transform(scores.to_frame())
        #     data.loc[scores_selector, column] = scores_feat

        column = f"score_maxabs_{guess_id}"
        data.loc[scores_unknown_selector, column] = scores_unknown
        if scores_selector_sum > 0:
            scores_feat = preprocessing.MaxAbsScaler().fit_transform(scores.to_frame())
            data.loc[scores_selector, column] = scores_feat

        column = f"rank_std_{guess_id}"
        data.loc[scores_unknown_selector, column] = scores_unknown
        if scores_selector_sum > 0:
            ranks_feat = preprocessing.StandardScaler().fit_transform(ranks.to_frame())
            data.loc[scores_selector, column] = ranks_feat

        column = f"rank_maxabs_{guess_id}"
        data.loc[scores_unknown_selector, column] = scores_unknown
        if scores_selector_sum > 0:
            ranks_feat = preprocessing.MaxAbsScaler().fit_transform(ranks.to_frame())
            data.loc[scores_selector, column] = ranks_feat

# Instead of relying on setting this value, it's (quite) safe to calculate the
# number of added features.
num_features = (len(data.columns) - NUM_INFO_COLUMNS -
                num_guessers*NUM_COLS_PER_GUESSER) / num_guessers
assert num_features == int(num_features)
num_features = int(num_features)

range_info_columns = list(range(NUM_INFO_COLUMNS))
# range_info_columns = [0, 2, 4]    # only a selection of list(range(NUM_INFO_COLUMNS))
range_org_columns_suggested = np.array([NUM_INFO_COLUMNS+gid*NUM_COLS_PER_GUESSER for gid in guesser_ids])
range_org_columns_score = range_org_columns_suggested + 1
range_org_columns_rank = range_org_columns_suggested + 2
range_feats = np.array([NUM_INFO_COLUMNS+NUM_COLS_PER_GUESSER*num_guessers+gid*num_features for gid in guesser_ids])
range_feats = [elem for fid in range(num_features) for elem in list(range_feats + fid)]

guesser_columns = [data.columns[cid] for cid in sorted(
    list(range_info_columns) + list(range_org_columns_suggested) +
    list(range_org_columns_score) + list(range_org_columns_rank) + list(range_feats))]

data.to_csv(sys.stdout, index=False, sep="\t", na_rep=UNKNOWN_VALUE,
            columns=guesser_columns)
