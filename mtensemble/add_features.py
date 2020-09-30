#!/usr/bin/env python3
"""
Calculate (per error id) and add features
 - delta values for confidence columns of each algorithm
 - normalised confidence values



 we cannot just run it with new random splits because errors are in different subsets?
 so prob one error should be in one new fold
"""

from sys import stdin
import sys
import numpy as np
from sklearn import preprocessing

NUM_ALGORITHMS = 5

nn_data = dict()
header = ""
#error_id	type	error_length	suggestion	is_correct	10_gram_is_suggested	10_gram_score	1_gram_is_suggested
# 1_gram_score	3_gram_is_suggested	3_gram_score	5_gram_is_suggested	spellcheker_suggested	spellcheker_score
#
keys = ['error_id', 'type', 'error_length', 'suggestion', 'is_correct', '10_pred', '10_score', '1_pred', '1_score',
        '3_pred', '3_score', '5_pred', '5_score', 'spell_pred', 'spell_score']
# keys = ['error_id', 'type', 'error_length', 'suggestion', 'is_correct', '10_pred', '10_score', '1_pred', '1_score',
#         '5_pred', '5_score', 'spell_pred', 'spell_score']

for line_id, line in enumerate(stdin):
    line = line.strip()
    # save header for later
    if line_id == 0:
        header = line.split('\t')
        # header = '\t'.join(header[:7] + header[9:13])
        # header = '\t'.join(header[:9] + header[11:])
        header = '\t'.join(header[:])
        continue
    # split data fields for processing...
    fields = [s.strip() for s in line.split('\t')]
    error_id = fields[0]
    # ...and put all lines for one key:error_id into a value:list
    if error_id not in nn_data:
        # nn_data[error_id] = list()
        nn_data[error_id] = dict()
    nn_data[error_id].update(zip(keys, fields))
    # try:
    for k, v in nn_data[error_id].items():
        # if 'score' in k:
        if 'score' in k:
            nn_data[error_id][k] = float(v)
        if 'pred' in k:
        # if 'pred' in k:
            nn_data[error_id][k] = int(v)

# output header
print(header + "\t" + "\t".join(["conf_norm_"+str(x) for x in
                                 range(NUM_ALGORITHMS)] +
                                ["delta_"+str(x) for x in
                                 range(NUM_ALGORITHMS)]))

# sys.stderr.write(str(nn_data))
# sys.exit()


for error_id in nn_data:
    # the field ids we want to calculate the deltas for:
    # 5,7,9,11,...
    # conf_ids = list(range(5, (4+2*NUM_ALGORITHMS), 2))
    # sys.stderr.write(str(conf_ids))
    _confvals = []
    _classes = []
    sugg_dict = nn_data[error_id]
    # for corr in nn_data[error_id]:
    # sys.stderr.write(str(sugg_dict))
    confval_list = []
    classes_list = []
    for k, v in sugg_dict.items():
        if 'score' in k:
            confval_list.append(v)
        elif 'pred' in k:
            classes_list.append(v)
    _confvals.append(confval_list)
    _classes.append(classes_list)
    # _confvals.append([float(corr[conf_id]) for conf_id in conf_ids])
    # _classes.append([int(corr[conf_id-1]) for conf_id in conf_ids])
    sys.stderr.write(str(_confvals))

    # sys.stderr.write(str(_classes))
    # NumPy comes in handy here
    confvals = np.asarray(_confvals)
    classes = np.asarray(_classes)
    # get the min/max confidence values per column
    #confvals_maxs = np.amax(confvals, axis=0)
    confvals_mins = np.amin(confvals, axis=0)
    # print(confvals_mins)
    ## stack these 1d-arrays to get a matrix of the same size as the others
    confvals_mins_stack = np.repeat([confvals_mins], np.size(confvals, axis=0),
                                    axis=0)
    # get the True/False matrix for selecting confidence values of "-1" and "0"
    # classes
    classes_minus_ones = np.equal(classes, -1)
    classes_zeros = np.equal(classes, 0)

    # DEBUG:
    # print(confvals)
    #print(classes)
    #print(confvals_mins)
    # print(confvals_mins_stack/2)
    # print(classes_minus_ones)

    # CALCULATE:
    #  * class: -1 (i did not propose this correction)
    #  * confval: my best suported correction
    # to
    #  * class: -1
    #  * confval: min of all supported corrections for this error_id
    confvals[classes_minus_ones] = confvals_mins_stack[classes_minus_ones]
    # print(confvals)
    # CALCULATE:
    #  - normalised confvals per error_id
    #  - deltas of these normalise confidence values
    confvals_norm = preprocessing.normalize(confvals, norm='l2', axis=0)

    # print(confvals_norm)
    deltas = 1 - confvals_norm
    # print(deltas)
    # output data
    confvals_norm_list = confvals_norm.tolist()
    deltas_list = deltas.tolist()

    # sys.stderr.write(str(confvals_norm_list))
    # sys.stderr.write(str(deltas_list))
    assert len(confvals_norm_list) == 1
    assert len(deltas_list) == 1
    line = []
    for key in keys:
        line.append(sugg_dict[key])
    confvals_norm_line = [str(x) for x in confvals_norm_list[0]]
    deltas_line = [str(x) for x in deltas_list[0]]
    line += confvals_norm_line
    line += deltas_line
    line = [str(el) for el in line]
    sys.stderr.write(str(line))
    print("\t".join(line))

    # for corr_id, corr in enumerate(nn_data[error_id]):
    #     confvals_norm_line = [str(x) for x in confvals_norm_list[corr_id]]
    #     deltas_line = [str(x) for x in deltas_list[corr_id]]
    #
    #     line = corr[0] + [error_id] + corr[1:(4+(2*NUM_ALGORITHMS))]
    #     line += confvals_norm_line
    #     line += deltas_line
    #     print("\t".join(line))