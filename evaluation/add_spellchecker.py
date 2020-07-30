# -*- coding: utf-8 -*-

import os
import codecs
import re
import csv
import sys
import pickle
from os import listdir
from os.path import isfile, join
import pandas as pd
import nltk
import nltk.translate.gleu_score as gleu
import ast
from collections import OrderedDict
from collections import namedtuple
import random
import numpy as np
import string
from spellchecker import SpellChecker
import Levenshtein


data_dir = '/Users/katinska/GramCorr/mtensemble/input/new_folds_last'
out_data_dir = '/Users/katinska/GramCorr/mtensemble/input/folds_with_spell'


def add_spellchecker_scores(input):
    spell = SpellChecker(language='de')
    n = 0
    s = 0
    with codecs.open(os.path.join(data_dir, input), mode='r') as table_file:
        with codecs.open(os.path.join(out_data_dir, input), mode='w') as full_out:
            table_reader = csv.reader(table_file, delimiter='\t')
            writer = csv.writer(full_out, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for i, row in enumerate(table_reader):
                # print(row)
                if i == 0:
                    header = row
                    writer.writerow(header)
                else:
                    expected = row[0]
                    # suggestion = row[4]
                    # print(expected)
                    # print(suggestion)
                    error = row[0].split('_')[-1]
                    print('Error: >>>>>>>>', error)
                    misspelled = spell.unknown([error])
                    suggections = []
                    n += 1
                    for w in misspelled:
                        suggections.append(spell.correction(w))
                    distanses = []
                    suggections = suggections[:5]
                    print('Suggestions', suggections)
                    if len(suggections):
                        s += 1
                    # for s in suggections:
                    #     distanses.append(spell.edit_distance_2(s))
                    # distanses = distanses[:5]
                    # print('Distances', distanses)

    print('Errors: ', n)
    print('Suggested:', s)
            # sys.exit()
    return True

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


def get_suggections():
    suggested = codecs.open(os.path.join(out_data_dir, 'hunspell_corrections.txt'), mode='r')
    filtered_lines = []
    for sline in suggested:
        print(sline.strip())
        if sline.strip() is None:
            continue
        else:
            filtered_lines.append(sline)

    # print(len(filtered_lines))
    suggestion_dict = dict()
    for s in filtered_lines:
        if s.startswith('&'):
            split = s.split(':')
            error = split[0].split(' ')[1].strip()
            ss = split[1].split(',')
            if error not in suggestion_dict:
                suggestion_dict[error] = [el.strip() for el in ss]

    # print(suggestion_dict)
    print(len(suggestion_dict))
    for k, v in suggestion_dict.items():
        suggestion_dict[k] = v[:5]

    files = os.listdir(data_dir)
    print(files)
    for file in files:
        if file.endswith('.csv') and 'fold' in file:
            print(file)
            with codecs.open(os.path.join(data_dir, file), mode='r') as table_file:
                with codecs.open(os.path.join(out_data_dir, file), mode='w') as full_out:
                    table_reader = csv.reader(table_file, delimiter='\t')
                    writer = csv.writer(full_out, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    for i, row in enumerate(table_reader):
                        if i == 0:
                            header = row
                            header.append('spellcheker_suggested')
                            header.append('spellcheker_score')
                            writer.writerow(header)
                        else:
                            print(i)
                            if '2' in file:
                                print(row)
                            expected = row[0]
                            error = row[1].split('_')[-1]
                            if error in suggestion_dict:
                                for e in suggestion_dict[error]:
                                    new_line = row
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
                                writer.writerow(new_line)
                            else:
                                row.extend(['0']*2)
                                writer.writerow(row)


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