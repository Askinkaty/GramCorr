#!/usr/bin/env python3

import argparse
import logging
import sys

import pandas as pd
import numpy as np
from sklearn import preprocessing

NUM_INFO_COLUMNS = 5    # Fix the number of 'info' columns
UNKNOWN_VALUE = "?"     # Set the character to represent missing/unknown values


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
data = pd.read_csv(sys.stdin, sep='\t', na_values="?", header=0)

num_guessers = 5

# Set the guesser_ids to use when writing the output
if args.guesser_ids:
    guesser_ids = [int(gid.strip()) for gid in args.guesser_ids.split(',')]
else:
    guesser_ids = list(range(num_guessers))
assert max(guesser_ids) <= num_guessers

# Instead of relying on setting this value, it's (quite) safe to calculate the
# number of added features.
num_features = (len(data.columns) - NUM_INFO_COLUMNS - num_guessers*2) / num_guessers
assert num_features == int(num_features)
num_features = int(num_features)

range_info_columns = list(range(NUM_INFO_COLUMNS))
range_org_columns_suggested = np.array([NUM_INFO_COLUMNS+gid*2 for gid in guesser_ids])
range_org_columns_score = range_org_columns_suggested + 1
range_feats = np.array([NUM_INFO_COLUMNS+2*num_guessers+gid*num_features for gid in guesser_ids])
range_feats = [elem for fid in range(num_features) for elem in list(range_feats + fid)]

guesser_columns = [data.columns[cid] for cid in sorted(list(range_info_columns) +
                         list(range_org_columns_suggested) +
                         list(range_org_columns_score) + list(range_feats))]

data[(data.iloc[:, range_org_columns_suggested] != 0).any(axis=1)].to_csv(
    sys.stdout, index=False, sep="\t", na_rep=UNKNOWN_VALUE,
    columns=guesser_columns)
