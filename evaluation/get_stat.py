# -*- coding: utf-8 -*-

import os
import codecs
import re
import csv
import numpy as np
import pickle
import collections

import sys
import random

from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, plot_roc_curve, roc_auc_score

data_dir = '/Users/katinska/GramCorr/mtensemble/input/folds_with_spell'



def correlation():
    model_data = {'10': [], '1': [], '3': [], '5': [], 'spell': []}
    files = os.listdir(data_dir)
    for file in files:
        if file.endswith('.csv') and 'fold' in file:
            with codecs.open(os.path.join(data_dir, file), mode='r') as table_file:
                table_reader = csv.reader(table_file, delimiter='\t')
                for i, row in enumerate(table_reader):
                    # error_id	type	error_length	suggestion	is_correct	10_gram_is_suggested	10_gram_score
                    # 	1_gram_is_suggested	1_gram_score	3_gram_is_suggested	3_gram_score
                    # 5_gram_is_suggested	5_gram_score	spellcheker_suggested	spellcheker_score
                    if i == 0:
                        continue
                    else:
                        model_data['10'].append(row[5])
                        model_data['1'].append(row[7])
                        model_data['3'].append(row[9])
                        model_data['5'].append(row[11])
                        model_data['spell'].append(row[13])
    for k, v in model_data.items():
        model_data[k] = np.array(v).astype(np.float)
    m = np.vstack(model_data.values())
    cor = np.corrcoef(m)
    print(cor)


def collect_not_corrected():
    files = os.listdir(data_dir)
    for file in files:
        if file.endswith('.csv') and 'fold' in file:
            with codecs.open(os.path.join(data_dir, file), mode='r') as table_file:
                table_reader = csv.reader(table_file, delimiter='\t')
                for i, row in enumerate(table_reader):
                    if i == 0:
                        continue
                    else:
                        if all([el != '1' for el in [row[5], row[7], row[9], row[11], row[13]]]):
                            print('not corrected')


def get_not_corrected_error_dict():
    dict_path = '/Users/katinska/GramCorr/evaluation/error_dict_check2.pkl'
    original_path = '/Users/katinska/GramCorr/corpora/LearnerCorpora/Koko/Koko'
    out_file = '/Users/katinska/GramCorr/evaluation/out.txt'
    out = codecs.open(out_file, 'w', encoding='utf-8')
    files = os.listdir(original_path)
    out_not_corrected_by_smt = '/Users/katinska/GramCorr/evaluation/types_smt.txt'
    out_smt = codecs.open(out_not_corrected_by_smt, 'w', encoding='utf-8')
    random_errors = []
    error_types = dict()
    with open(dict_path, 'rb') as pickle_file:
        error_data = pickle.load(pickle_file)
        print(len(error_data.keys()))
        all_errors = len(error_data.keys())
        corrected = 0
        not_corrected = 0
        for error, d in error_data.items():
            corrected_error = False
            c = []
            for model_key, m_dict in d.items():
                if 'gram' not in model_key:
                    continue
                c.append(m_dict.get('corrected'))
                if m_dict.get('corrected') == 1:
                    corrected_error = True
                    break
            if any([el == 1 for el in c]):
                corrected += 1
            else:
                not_corrected += 1
            if d['type'][0] not in error_types:
                error_types[d['type'][0]] = [1, int(corrected_error)]
            else:
                error_types[d['type'][0]][0] += 1
                error_types[d['type'][0]][1] += int(corrected_error)

            if not corrected_error:
                random_errors.append(error)

    random.shuffle(random_errors)
    # for er in random_errors:
    #     if random.random() < 0.5 and len(random_examples) < 300:
    #         random_examples.append(er)
    random_examples = random_errors
    print(f'Not corrected %: {corrected/all_errors}')
    print(f'Corrected %: {not_corrected/all_errors}')
    print(f'Corrected #: {corrected}')
    print(f'Not corrected #: {not_corrected}')
    print(f'All errors #: {all_errors}')
    for k, v in error_types.items():
        error_types[k] = [round(v[1]/v[0], 2), v[1], v[0]]
    print(error_types)

    sorted_error_types = {k: v for k, v in sorted(error_types.items(), key=lambda item: item[1])}
    sorted_error_types = collections.OrderedDict(sorted_error_types)

    for k, v in sorted_error_types.items():
        out_smt.write(k + ' : ' + str(v[0]) + ', ' + str(v[1]) + '/' + str(v[2]) + '\n')

    # print(random_examples)
    # lines = []
    # for er_id in random_examples:
    #     split = er_id.split('_')
    #     file_name, line_number = split[2], split[3]
    #     for file in files:
    #         if file == file_name:
    #             with codecs.open(os.path.join(original_path, file), 'r') as f:
    #                 original = f.readlines()
    #                 for i, line in enumerate(original):
    #                     if int(i) == int(line_number) and 'error type' in line:
    #                         lines.append(line)

    # print(lines)
    # print(len(lines))
    # for l in lines:
    #     out.write(l)

# import scikitplot as skplt


def get_errors_ensemble():
    only_other_hypotheses = True
    # dict_path = '/Users/katinska/GramCorr/evaluation/error_dict_check2.pkl'
    # with open(dict_path, 'rb') as pickle_file:
    #     error_data = pickle.load(pickle_file)
    #     for key, value in error_data.items():
    #         print(key)
    #         print(value)
    #         sys.exit()

    file_path = '/Users/katinska/GramCorr/mtensemble/output/output_exp'
    original_path = '/Users/katinska/GramCorr/corpora/LearnerCorpora/Koko/Koko'

    # input_path = '/Users/katinska/GramCorr/mtensemble/input/features'
    input_path = '/Users/katinska/GramCorr/mtensemble/input/new_folds_last_1'

    # out_file = '/Users/katinska/GramCorr/evaluation/out_ensemble_other_hyp.txt'
    out_not_corrected_by_ensemble = '/Users/katinska/GramCorr/evaluation/types_ensemble_10_1_5_spell_gram.txt'
    # out = codecs.open(out_file, 'w', encoding='utf-8')
    out_ensemble = codecs.open(out_not_corrected_by_ensemble, 'w', encoding='utf-8')
    files = os.listdir(original_path)
    pred_files = os.listdir(file_path)
    input_files = os.listdir(input_path)
    random_examples = []
    random_errors = []
    incorrect = 0
    correct = 0
    all = 0
    valid_provided = 0
    invalid_provided = 0
    error_types = dict()
    er_dict = {}
    suggestion_dict = dict()
    average_score = []
    average_error_score = []
    y_true = []
    y_score = []
    for f in input_files:
        if f.endswith('.csv'):
            input_file = os.path.join(input_path, f)
            with codecs.open(input_file, mode='r') as table_file:
                table_reader = csv.reader(table_file, delimiter='\t')
                for j, row in enumerate(table_reader):
                    if j == 0:
                        continue
                    error = row[0].strip().split('_')[-1]
                    suggestion = row[3].strip()
                    expected = row[-1].strip()
                    if suggestion != error and suggestion != expected:
                        scores = [row[5]]
                        er_dict[row[0].strip()] = [scores, row[1]]
                        suggestion_dict[row[0].strip()] = suggestion
    for i, pred_file in enumerate(pred_files):
        if 'pred' in pred_file:
            pf = codecs.open(os.path.join(file_path, pred_file), 'r', encoding='utf-8')
            lines = pf.readlines()
            for line in lines[1:]:
                corrected = False
                split = line.split(',')
                if len(split) > 1:
                    id = split[5].strip()
                else:
                    continue
                if split[3] == '+': # error
                    y_score.append(1 - float(split[4]))
                    incorrect += 1
                    average_error_score.append(float(split[4]))
                    if id in er_dict: # check that this error is not the same as expected or given answer
                        random_errors.append(id)
                else:
                    y_score.append(float(split[4]))
                    correct += 1
                    corrected = True
                y_true.append(1)
                average_score.append(float(split[4]))
                all += 1
                if er_dict.get(id):
                    if er_dict[id][1] not in error_types:
                        error_types[er_dict[id][1]] = [1, int(corrected)]
                    else:
                        error_types[er_dict[id][1]][0] += 1
                        error_types[er_dict[id][1]][1] += int(corrected)
                try:
                    if any([s == '1' for s in er_dict[id][0]]):
                        valid_provided += 1
                    else:
                        invalid_provided += 1
                except:
                    pass
    random.shuffle(random_errors)
    for er in random_errors:
        if random.random() < 0.5 and len(random_examples) < 500:
            random_examples.append(er)
    error_lines = []
    for er_id in random_examples:
        # print('er id: ', er_id)
        er_split = er_id.split('_')
        s = suggestion_dict[er_id]
        # print(s)
        if not s:
            continue
        file_name, line_number, ind, error = er_split[2], er_split[3], int(er_split[4]), er_split[-1]
        # print(error)

        for file in files:
            if file == file_name:
                with codecs.open(os.path.join(original_path, file), 'r') as f:
                    original = f.readlines()
                    for i, line in enumerate(original):
                        if int(i + 1) == int(line_number) and 'error type' in line:

                            if line not in error_lines:
                                index = line.find(error, ind, len(line))
                                replaced = line[:index] + s + line[index + len(error):]
                                error_lines.append(replaced)

    # for l in error_lines:
    #     out.write(l)
    print(f'All: {all}')
    print(f'Correct: {correct}')
    print(f'Incorrect: {incorrect}')
    print(f'At least 1 valid sugg: {valid_provided}')
    print(f'No valid suggestions: {invalid_provided}')
    print(f'Average score: {sum(average_score)/len(average_score)}')
    print(f'Average error score: {sum(average_error_score)/len(average_error_score)}')
    print(np.mean(average_score))
    print(np.mean(average_error_score))
    print(f'Variance: {np.var(average_score)}')
    print(f'0.95 percentile: {np.percentile(average_score, 95)}')
    print(f'0.05 percentile: {np.percentile(average_score, 5)}')

    print(f'Error score variance: {np.var(average_error_score)}')
    print(f'Error 0.95 percentile: {np.percentile(average_error_score, 95)}')
    print(f'Error 0.05 percentile: {np.percentile(average_error_score, 5)}')
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    print(average_error_score)
    print(average_score)
    # fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)
    # print(f'Thresholds: {thresholds}')

    # plot_roc_curve(fpr, tpr)
    # auc = roc_auc_score(y_true, y_score)
    # print(f'AUC: {auc}')
    # print(error_types)
    for k, v in error_types.items():
        error_types[k] = [round(v[1]/v[0], 2), v[1], v[0]]
    # print(error_types)
    sorted_error_types = {k: v for k, v in sorted(error_types.items(), key=lambda item: item[1])}
    sorted_error_types = collections.OrderedDict(sorted_error_types)

    for k, v in sorted_error_types.items():
        # print(k, v)
        out_ensemble.write(k + ' : ' + str(v[0]) + ', ' + str(v[1]) + '/' + str(v[2]) + '\n')

if __name__ == '__main__':
    # correlation()
    # collect_not_corrected()
    # get_not_corrected_error_dict()
    get_errors_ensemble()