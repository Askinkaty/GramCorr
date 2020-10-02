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
import numpy as np
from collections import OrderedDict


error_table_file = '../corpora/LearnerCorpora/Koko/cv/error_coordinates.csv'
processed = '../translate/Koko/split_processed/'


def get_index(full_line):
    result = []
    prev = 0
    for l in full_line:
        print('LINE', '>'+ l + '<')
        if l.startswith('$ $ $'):
            l = l.replace('$ $ $', '').lstrip()
            error = l
            ind = prev
            print('BELLO', ind)
            result.append((ind, error.strip()))
            # if l.strip():
            prev += len(error)
        else:
            print('1', len(l))
            print('2', prev)
            prev += len(l)
            print('PREV', prev)
    return result


if __name__== '__main__':
    err_table = pd.read_csv(error_table_file)
    err_table['fold#'] = err_table['fold#'].apply(lambda x: int(re.sub(r'fold', '', x)))
    err_table.fillna('', inplace=True)
    err_table['new_ind'] = 0
    err_table['new_errind'] = 0
    folds = [dir for dir in os.listdir(processed) if os.path.isdir(processed + dir)]

    for fold in folds:
        s_lines = []
        t_lines = []
        cur_line = 1
        number = int(re.search(r'.*([0-9]+)', fold).group(1))
        source_file = os.path.join(processed, fold + '/train.en')
        target_file = os.path.join(processed, fold + '/train.de')
        source = codecs.open(source_file, 'r', encoding='utf-8') # error
        target = codecs.open(target_file, 'r', encoding='utf-8') # reference
        z = zip(source, target)
        full_s_line = []
        full_t_line = []
        for i, pair in enumerate(z):
            s_line = pair[0]
            t_line = pair[1]
            if '# # #' in s_line:
                s_lines.append((number, full_s_line))
                t_lines.append((number, full_t_line))
                full_s_line = []
                full_t_line = []
                continue
            else:

                full_s_line.append(s_line)
                full_t_line.append(t_line)



        for k, p in enumerate(zip(s_lines, t_lines)):
            if k > 200:
                err_table.to_csv('new_error_coordinates.csv')
                sys.exit()

            n = p[0][0]
            s = p[0][1]
            t = p[1][1]
            print(s)
            print(t)
            assert len(s) == len(t)
            if len(s) > 1:
                ind_er = get_index(s)
                ind_cor = get_index(t)
                print(ind_er)
                print(ind_cor)
                if ind_cor and ind_er:
                    assert len(ind_er) == len(ind_cor)
                    for j, el in enumerate(ind_er):
                        print(cur_line)
                        errors = err_table.loc[(err_table['fold#'] == n) & (err_table['line'] == cur_line)]
                        i = err_table.loc[(err_table['fold#'] == n) & (err_table['line'] == cur_line)
                                          & (err_table['error'] == ind_er[j][1])].index.values.tolist()
                        print(i)
                        if i:
                            i = i[0]
                            print('IND to fix', i)
                            print(ind_er[j])
                            print(ind_cor[j])
                            err_table.loc[i, 'new_ind'] = ind_cor[j][0]
                            err_table.loc[i, 'new_errind'] = ind_er[j][0]
                            print(err_table.loc[i,:])



            cur_line += 1
    err_table.to_csv('new_error_coordinates.csv')
