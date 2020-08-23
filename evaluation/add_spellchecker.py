# -*- coding: utf-8 -*-

import os
import codecs
import re
import csv
import sys
# import pickle
# from os import listdir
# from os.path import isfile, join
# import pandas as pd
# import nltk
# import nltk.translate.gleu_score as gleu
# import ast
# from collections import OrderedDict
# from collections import namedtuple
# import random
# import numpy as np
# import string
# from spellchecker import SpellChecker
import Levenshtein
import random


data_dir = '/Users/katinska/GramCorr/mtensemble/input/new_folds_last'
out_data_dir = '/Users/katinska/GramCorr/mtensemble/input/folds_with_spell'

def collect_errors(file):
    if file == 'data.csv':
        words = []
        with codecs.open(os.path.join(data_dir, file), mode='r') as table_file:
            with codecs.open(os.path.join(out_data_dir, file.replace('csv', 'txt')), mode='w') as full_out:
                table_reader = csv.reader(table_file, delimiter='\t')
                for i, row in enumerate(table_reader):
                    if i == 0:
                        header = row
                    else:
                        expected = row[1]
                        error = row[1].split('_')[-1]
                        if error not in words:
                            words.append(error)

                for e in words:
                    full_out.write(e + '\n')


def split_list(all_rows, n):
    all_folds = []
    av_len = len(all_rows)/n
    for i in range(n):
        new_fold = []
        if i != n - 1:
            while len(new_fold) < int(av_len):
                new_fold.append(all_rows[-1])
                all_rows = all_rows[:-1]
            all_folds.append(new_fold)
        else:
            all_folds.append(all_rows)
    return all_folds


def get_hunspell_suggestions():
    suggested = codecs.open(os.path.join(out_data_dir, 'hunspell_corrections.txt'), mode='r')
    filtered_lines = []
    for sline in suggested:
        print(sline.strip())
        if sline.strip() is None:
            continue
        else:
            filtered_lines.append(sline)

    suggestion_dict = dict()
    for s in filtered_lines:
        if s.startswith('&'):
            split = s.split(':')
            error = split[0].split(' ')[1].strip()
            ss = split[1].split(',')
            if error not in suggestion_dict:
                suggestion_dict[error] = [el.strip() for el in ss]
    print(len(suggestion_dict))
    for k, v in suggestion_dict.items():
        suggestion_dict[k] = v[:5]
    return suggestion_dict


def get_suggections():
    suggestion_dict = get_hunspell_suggestions()
    files = os.listdir(data_dir)
    all_rows = []
    header = None
    for file in files:
        if file.endswith('.csv') and 'fold' in file:
            print(file)
            with codecs.open(os.path.join(data_dir, file), mode='r') as table_file:
                table_reader = csv.reader(table_file, delimiter='\t')
                for i, row in enumerate(table_reader):
                    if i == 0:
                        header = row
                        header.append('spellcheker_suggested')
                        header.append('spellcheker_score')
                    else:
                        print(i)
                        expected = row[0]
                        error = row[1].split('_')[-1]
                        if error in suggestion_dict:
                            for e in suggestion_dict[error]:
                                new_line = row[:]
                                if e == expected:
                                    new_line[4] = e
                                    new_line[5] = '1'
                                    new_line.append('1')
                                else:
                                    new_line[4] = e
                                    new_line[5] = '0'
                                    new_line.append('-1')
                                distance = Levenshtein.distance(e, expected)
                                new_line.append(distance)
                                new_line = new_line[1:]
                                all_rows.append(new_line)
                        else:
                            row.extend(['0']*2)
                            row = row[1:]
                            all_rows.append(row)
    random.shuffle(all_rows)
    all_folds = split_list(all_rows, len(files))
    for i, file in enumerate(files):
        if file.endswith('.csv') and 'fold' in file:
            with codecs.open(os.path.join(out_data_dir, file), mode='w') as full_out:
                writer = csv.writer(full_out, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(header[1:])
                for el in all_folds[i]:
                    writer.writerow(el)



if __name__ == '__main__':
    # path_to_errors = 'error_dict.pkl'
    # with open(path_to_errors, 'rb') as pickle_file:
    #     error_data = pickle.load(pickle_file)
    #     for key, value in error_data.items():
    #         print(value)
    #         print(key)
    # sys.exit()

    # files = os.listdir(data_dir)
    # for file in files:
    #     if file.endswith('.csv'):
    #         print(file)
    #         add_spellchecker_scores(file)
            # collect_errors(file)

    get_suggections()