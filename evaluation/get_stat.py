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


# from sklearn import metrics
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, auc, plot_roc_curve, roc_auc_score

data_dir = '/Users/katinska/GramCorr/mtensemble/input/folds_with_spell_1'



def correlation():
    model_data = {'10': [], '1': [], '3': [], '5': [], 'spell': []}
    model_data = collections.OrderedDict(model_data)
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
                        print(row[5], row[7], row[9], row[11], row[13])
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

#               10-char    1-gram         3-gram          5-gram         spell
# 10-char  [[ 1.         -0.24018704 -0.20670782 -0.2037281   0.31693091]
# 1-gram    [-0.24018704  1.          0.28393159  0.26397599  0.0077318 ]
# 3-gram    [-0.20670782  0.28393159  1.          0.72559986 -0.01179528]
# 5-gram    [-0.2037281   0.26397599  0.72559986  1.         -0.01158216]
# spell        [ 0.31693091  0.0077318  -0.01179528 -0.01158216  1.        ]]




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
    input_path = '/Users/katinska/GramCorr/mtensemble/input/new_folds_last_2'

    # out_file = '/Users/katinska/GramCorr/evaluation/out_ensemble_other_hyp.txt'
    out_not_corrected_by_ensemble = '/Users/katinska/GramCorr/evaluation/types_ensemble_all.txt'
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

import json
import pprint

import csv

def get_type_dict():
    dir = '/Users/katinska/GramCorr/mtensemble/input/folds_with_spell_2'
    type_dict = dict()
    for file in os.listdir(dir):
        if not file.endswith('csv'):
            continue
        reader = csv.DictReader(open(os.path.join(dir, file)), delimiter='\t')
        for line in reader:
            if line['type'] not in type_dict:
                type_dict[line['type']] = []
                type_dict[line['type']].append(line)
            else:
                type_dict[line['type']].append(line)
    return type_dict



def group_by_err_id(d):
    new_type_dict = dict()
    for k, v in d.items():
        new_type_dict[k] = dict()
        for el in v:
            if el['error_id'] not in new_type_dict[k]:
                new_type_dict[k][el['error_id']] = []
                new_type_dict[k][el['error_id']].append(el)
            else:
                new_type_dict[k][el['error_id']].append(el)
    return new_type_dict


def create_model_dict():
    keys = ['1_gram', '10_gram', '3_gram', '5_gram', 'spell']
    d = dict()
    for k in keys:
        d[k] = {'corrected': 0, 'touched': 0}
    return d

def get_stat_for_type(d):
    final_dict = dict()
    for type, v in d.items():
        final_dict[type] = dict()
        for err_id, e_v in v.items():
            model_dict = create_model_dict()
            for sugg in e_v:
                correct = sugg['is_correct']
                for mk, v in model_dict.items():
                    for sk, sd in sugg.items():
                        if 'suggested' in sk and sk.startswith(mk):
                            if sugg[sk] == '1' and correct == '1':
                                model_dict[mk]['corrected'] = 1
                                model_dict[mk]['touched'] = 1
                                continue
                            elif sugg[sk] == '1' and correct == '0':
                                model_dict[mk]['touched'] = 1
                                continue
                            elif sugg[sk] == '-1':
                                model_dict[mk]['touched'] = 1
                                continue
            final_dict[type][err_id] = model_dict

    return final_dict


def get_total_type():
    table = '/Users/katinska/GramCorr/evaluation/out_moses_table_last12.csv'
    type_dict = dict()
    type_num = dict()
    with codecs.open(table) as table_file:
        table_reader = csv.reader(table_file, delimiter='\t')
        for row in table_reader:
            error = row[0]
            b = error.split('_')[:-1]
            error = '_'.join(b)
            type = row[1]
            if error not in type_dict:
                type_dict[error] = type
            if type not in type_num:
                type_num[type] = 1
            elif type in type_num:
                type_num[type] += 1
    return type_num


def aggregate_type_stat(d):
    new_type_dict = dict()
    for type, v in d.items():
        model_dict = create_model_dict()
        for er_id, ev in v.items():
            for mk, mv in ev.items():
                model_dict[mk]['corrected'] += mv['corrected']
                model_dict[mk]['touched'] += mv['touched']
        new_type_dict[type] = model_dict
    return new_type_dict

def get_f_measure(p, r):
    if (p + r) != 0:
        f_m = round(2 * ((p * r) / (p + r)), 2)
    else:
        f_m = 0.0
    return f_m

def prepare_err_info():
    types = ["orth: 02 lcp instead of cap: other cases", "orth: 11 omissions: of double consonants",
            "orth: 06 sep instead of tog: compounds", "orth: 08 tog instead of sep: minimal phraseologism",
             "orth: 03 cap instead of lcp", "orth: 14 omissions: of one grapheme",
             "orth: 28 Eigennamen", "orth: 07 sep instead of tog: other cases",
             "orth: 09 tog instead of sep: other cases", "orth: 15 insertions: of double consonants",
             "corr: case"]
    type_dict = get_type_dict()
    aggregated_type_dict = group_by_err_id(type_dict)
    stat_dict = get_stat_for_type(aggregated_type_dict)
    type_total = get_total_type()

    final_stat_dict = aggregate_type_stat(stat_dict)
    other_model = create_model_dict()
    other_total = 0

    for type, v in final_stat_dict.items():
        if type in types:
            print('*'*100)
            print(f'TYPE: {type}')
            for mk, mv in v.items():
                print(f'Model: {mk}')
                if type in type_total:
                    total = type_total[type]
                    if mv['touched'] != 0:
                        p = round(mv['corrected'] / mv['touched'], 2)
                    else:
                        p = 0.0
                    r = round(mv['corrected'] / total, 2)
                    f_m = get_f_measure(p, r)
                    print(f'Type: {type}, P: {p}, R: {r}, F: {f_m}')
        else:
            if type in type_total:
                total = type_total[type]
                other_total += total
                for mk, mv in v.items():
                    other_model[mk]['corrected'] += mv['corrected']
                    other_model[mk]['touched'] += mv['touched']
    for k, v in other_model.items():
        print('Other types___________')
        print(f'MODEL: {k}')
        if other_model[k]['touched'] != 0:
            p = round(other_model[k]['corrected'] / other_model[k]['touched'], 2)
        else:
            p = 0.0
        r = round(other_model[k]['corrected'] / other_total, 2)
        f_m = get_f_measure(p, r)
        print(f'Other Types: {type}, P: {p}, R: {r}, F: {f_m}')


def check_guessers():
    dir = '/Users/katinska/GramCorr/mtensemble/input/folds_with_spell_2'
    out_f = '/Users/katinska/GramCorr/mtensemble/input/out1.jsonl'
    lines = []
    with codecs.open(out_f, 'w') as out:
        mp = {'10_gram': 5, '1_gram': 7, '3_gram': 9, '5_gram': 11, 'spell': 13}
        model_dict = dict()
        for k in mp.keys():
            model_dict[k] = dict()
        all = 0
        for file in os.listdir(dir):
            if not file.endswith('csv'):
                continue
            with codecs.open(os.path.join(dir, file), mode='r') as table:
                for i, line in enumerate(table):
                    if i == 0:
                        continue
                    if not line.strip():
                        continue
                    if line not in lines:
                        lines.append(line)
                    else:
                        continue
                    all += 1
                    data = line.split('\t')
                    type = data[1]
                    correct = data[4]
                    for m, v in model_dict.items():
                        if type not in v:
                            model_dict[m][type] = {'corr': 0, 'touched': 0, 'none': 0, 'total': 0}
                        if data[mp[m]] == '1' and correct == '1':
                            model_dict[m][type]['corr'] += 1
                            model_dict[m][type]['touched'] += 1
                        elif data[mp[m]] == '1' and correct == '0':
                            model_dict[m][type]['touched'] += 1
                        elif data[mp[m]] == '0':
                            model_dict[m][type]['none'] += 1
                        elif data[mp[m]] == '-1':
                            model_dict[m][type]['touched'] += 1
                        model_dict[m][type]['total'] += 1
                        assert model_dict[m][type]['touched'] >= model_dict[m][type]['corr']

        print(model_dict)
        for k, v in model_dict.items():
            v = {k: vl for k, vl in sorted(v.items(), key=lambda item: item[1]['total'])}
            json.dump({k: v}, out)
            out.write('\n')
        # out.write(model_dict)
        print(all)
        pprint.pprint(model_dict)


def check_type_statistics():
    # error_id
    # type
    # error_length
    # suggestion
    # is_correct
    # 10_gram_is_suggested 5
    # 10_gram_score
    # 1_gram_is_suggested 7
    # 1_gram_score
    # 3_gram_is_suggested 9
    # 3_gram_score
    # 5_gram_is_suggested 11
    # 5_gram_score
    # spellcheker_suggested
    # spellcheker_score

    # fold = ''
    # file = '/Users/katinska/GramCorr/corpora/LearnerCorpora/Koko/cv/error_coordicates_new.csv'
    dir = '/Users/katinska/GramCorr/mtensemble/input/folds_with_spell_1'
    for file in os.listdir(dir):
        if not file.endswith('csv'):
            continue
        with codecs.open(os.path.join(dir, file), mode='r') as table:
            stat_file = dict()
            stat_type = dict()
            model_dict = {'1_gram': 0, '3_gram': 0, '5_gram': 0, '10_gram': 0}
            for i, line in enumerate(table):
                if i == 0:
                    continue
                if not line.strip():
                    continue
                data = line.split('\t')
                # print(data)
                f = data[0].split('_')[0]
                fl = data[0].split('_')[2]
                type = data[1]
                # if fold != f and fold != '':
                #     fold = f
                #     print(f'FOLD: {fold}')
                #     type_sorted = {k: v for k, v in sorted(stat_type.items(), key=lambda x: x[1], reverse=True)}
                #     file_sorted = {k: v for k, v in sorted(stat_file.items(), key=lambda x: x[1], reverse=True)}
                #     print(model_dict)
                #     print(type_sorted)
                #     print(file_sorted)
                #     stat_file = dict()
                #     stat_type = dict()
                # elif fold == '':
                #     fold = f
                if fl not in stat_file:
                    stat_file[fl] = 1
                else:
                    stat_file[fl] += 1
                if type not in stat_type:
                    stat_type[type] = 1
                else:
                    stat_type[type] += 1
                if data[5] == '1':
                    model_dict['10_gram'] += 1
                if data[7] == '1':
                    model_dict['1_gram'] += 1
                if data[9] == '1':
                    model_dict['3_gram'] += 1
                if data[11] == '1':
                    model_dict['5_gram'] += 1

            print(f'FOLD: {file}')
            type_sorted = {k: v for k, v in sorted(stat_type.items(), key=lambda x: x[1], reverse=True)}
            file_sorted = {k: v for k, v in sorted(stat_file.items(), key=lambda x: x[1], reverse=True)}
            print(model_dict)
            print(type_sorted)
            print(file_sorted)


def check_broken_fold():
    file = '/Users/katinska/GramCorr/mtensemble/tmp.txt'
    corrected = 0
    all = 0
    a = 0
    c = 0
    with codecs.open(file, 'r', encoding='utf-8') as f:
        for line in f:
            a += 1
            print(line.split(','))
            split = line.strip().split(',')
            cl = split[-1]
            if cl == '1':
                c += 1
                all += 1
                if split[2] == '1' or split[4] == '1' or split[6] == '1' or split[10] == '1':
                    corrected += 1

    print(corrected)
    print(all)
    print(a)
    print(c)
    print((c * 100) / a)


def prepare_stat():
    types = dict()
    stat = dict()
    file = '/Users/katinska/GramCorr/mtensemble/input/out.jsonl'
    with codecs.open(file, mode='r') as data:
        for line in data:
            d = json.loads(line)
            for type_d in d.values():
                for k, v in type_d.items():
                    total = v['total']
                    if k not in types:
                        types[k] = total
                    else:
                        types[k] += total
            type_sorted = {k: v for k, v in sorted(types.items(), key=lambda x: x[1], reverse=True)}
            for model, model_d in d.items():
                if model not in stat:
                    stat[model] = dict()
                    stat[model]['Other'] = {'corr': 0, 'touched': 0, 'total': 0}
                    stat[model]['total_corr'] = 0
                    stat[model]['total_touched'] = 0
                    stat[model]['total_all'] = 0
                for j, t in enumerate(type_sorted.keys()):
                    stat[model]['total_corr'] += model_d[t]['corr']
                    stat[model]['total_touched'] += model_d[t]['touched']
                    stat[model]['total_all'] += model_d[t]['total']
                    if j <= 10:
                        stat[model][t] = dict()
                        if model_d[t]['touched'] == 0:
                            stat[model][t]['precision'] = 0.0
                        else:
                            stat[model][t]['precision'] = round(model_d[t]['corr'] / model_d[t]['touched'], 2)
                        stat[model][t]['recall'] = round(model_d[t]['corr'] / model_d[t]['total'], 2)
                        if (stat[model][t]['precision'] + stat[model][t]['recall']) != 0:
                            stat[model][t]['f'] = round(2 * ((stat[model][t]['precision'] * stat[model][t]['recall']) / (stat[model][t]['precision'] + stat[model][t]['recall'])), 2)
                        else:
                            stat[model][t]['f'] = 0.0
                    else:
                        stat[model]['Other']['corr'] += model_d[t]['corr']
                        stat[model]['Other']['touched'] += model_d[t]['touched']
                        stat[model]['Other']['total'] += model_d[t]['total']

    # pprint.pprint(stat)
    print(type_sorted)
    for k, v in stat.items():
        print(f'Model: {k}')
        for i, t in enumerate(type_sorted.keys()):
            if i <= 10:
                print(f'Type: {t}, value: {v[t]}')
        if v['Other']['touched'] == 0:
            other_prec = 0.0
        else:
            other_prec = round(v['Other']['corr'] / v['Other']['touched'], 2)
        other_rec = round(v['Other']['corr'] / v['Other']['total'], 2)
        if (other_prec + other_rec) != 0:
            f_m = round(2 * ((other_prec * other_rec ) / (other_prec + other_rec)), 2)
        else:
            f_m = 0.0
        print(f'Type: Other, other_precision: {other_prec}, other_recall: {other_rec}')
        total_prec = round(v['total_corr'] / v['total_touched'], 2)
        total_rec = round(v['total_corr'] / v['total_all'], 2)
        total_f = round(2 * ((total_prec * total_rec) / (total_prec + total_rec)), 2)
        print(f'Total precision: {total_prec}, Total recall: {total_rec}, Total F: {total_f}')



def rtf_performance():
    """
    ~TODO: weird small table on the page 7
    Do we need to calculate the number of all invalid candidates which were discarded?

    :return:
    """
    pred_dir = '/Users/katinska/GramCorr/mtensemble/output/output_exp'
    dir = '/Users/katinska/GramCorr/mtensemble/input/folds_with_spell_2'
    folds_files = [f for f in os.listdir(dir) if f.endswith('csv')]
    files = [el for el in os.listdir(pred_dir) if el.endswith('.pred')]
    incorrect_picked = 0
    incorrect_not_picked = 0
    correct_not_picked = 0
    correct_picked = 0
    total_errors = 0
    error_21 = 0
    error_12 = 0
    error_not_picked = 0
    error_picked = 0
    l = 0
    for file in files:
        pf = codecs.open(os.path.join(pred_dir, file), 'r', encoding='utf-8')
        lines = pf.readlines()
        for line in lines[1:]:
            if not line.strip():
                continue
            l += 1
            error = line.split(',')[-1].strip()
            total_errors += 1
            corr = '+' not in line.split(',')[3]
            if corr:
                if line.split(',')[1] == '2:1' and line.split(',')[2] == '2:1':
                    correct_picked += 1
                if line.split(',')[1] == '1:0' and line.split(',')[2] == '1:0':
                    error_not_picked += 1
                # for f in folds_files:
                #     with codecs.open(os.path.join(dir, f), mode='r') as table:
                #         for i, l in enumerate(table):
                #             if i == 0:
                #                 continue
                #             if not l.strip():
                #                 continue
                #             data = l.split('\t')
                #             if data[0].strip() == error:
                #                 if data[4] == '0':
                #                     incorrect_not_picked += 1
            else:
                # 1,1:0,2:1,+,0.93,fold9_2592_ID2942.txt_27_67_67_wenig
                if line.split(',')[1] == '1:0' and line.split(',')[2] == '2:1':
                    error_picked += 1
                elif line.split(',')[1] == '2:1' and line.split(',')[2] == '1:0':
                    correct_not_picked += 1
    print(f'Error picked: {error_picked}')
    print(f'Error not picked: {error_not_picked}')
    print(f'Correct picked: {correct_picked}')
    print(f'Correct not picked: {correct_not_picked}')
    print(f'Total errors: {total_errors}')
    print(l)
    # Incorrect picked: 1468 -- error which was labelled as correct (?)


    # Incorrect not picked: 42157 (because we have multiple suggestions per every error generated by SMT systems, only one is correct)
    # Correct picked: 6942
    # Correct not picked: 1468
    # Total errors: 8410




def count_type_perf_rf():
    pred_dir = '/Users/katinska/GramCorr/mtensemble/output/experiment/5_guessers-10_folds/0_1_2_3_4'
    table = '/Users/katinska/GramCorr/evaluation/out_moses_table_032021.csv'
    files = [el for el in os.listdir(pred_dir) if el.endswith('.pred')]
    types = ["orth: 02 lcp instead of cap: other cases", "orth: 11 omissions: of double consonants",
            "orth: 06 sep instead of tog: compounds", "orth: 08 tog instead of sep: minimal phraseologism",
             "orth: 03 cap instead of lcp", "orth: 14 omissions: of one grapheme",
             "orth: 28 Eigennamen", "orth: 07 sep instead of tog: other cases",
             "orth: 09 tog instead of sep: other cases", "orth: 15 insertions: of double consonants",
             "corr: case"]
    type_dict = dict()
    type_num = dict()
    with codecs.open(table) as table_file:
        table_reader = csv.reader(table_file, delimiter='\t')
        for row in table_reader:
            error = row[0]
            b = error.split('_')[:-1]
            error = '_'.join(b).replace("'", '')
            # error = error.replace(' ', '_').replace("'", '')
            # if 'fold7_3368_ID2587.txt_14_183_182' in error:
            #     print(error)
            #     print(row)
            type = row[1]
            if error not in type_dict:
                type_dict[error] = type
            if type not in type_num:
                type_num[type] = 1
            elif type in type_num:
                type_num[type] += 1
    # print(len(type_dict))
    # sys.exit()
    # print('*'*100)
    stat = dict()
    for t in type_num:
        if t not in stat:
            stat[t] = {'correct': 0, 'error': 0, 'all': 0}
    all_errors = []
    for file in files:
        pf = codecs.open(os.path.join(pred_dir, file), 'r', encoding='utf-8')
        lines = pf.readlines()
        for line in lines[1:]:
            if not line.strip():
                continue
            b = line.split(',')[-1].split('_')[:6]
            error_ens = '_'.join(b).replace("'", '')
            # print(b = line.split(',')[-1].split('_')[:5])
            # error_ens = line.split(',')[-1].strip().replace("'", '')
            # if 'fold2_357_ID1734.txt_18_80_81' in error_ens:
            #     print(error_ens)
            #     b = error_ens.split('_')[:-1]
            #     error_ens = '_'.join(b).replace("'", '')
            #     print(error_ens)

                # print(line)
                # sys.exit()
            # b = error_ens.split('_')[:-1]
            # error_ens = '_'.join(b).replace("'", '')
            # print(error_ens)
            if error_ens not in all_errors:
                all_errors.append(error_ens)
            # if error_ens in type_dict:
            type_er = type_dict[error_ens]
            if line.split(',')[3] != '+':
                stat[type_er]['correct'] += 1
            else:
                stat[type_er]['error'] += 1
            stat[type_er]['all'] += 1
    # print(stat)
    # print('*'*100)
    # for type in types:
    # print(len(all_errors))
    # sys.exit()
    for type in stat.keys():
        print(stat[type])
        print(type_num[type])
        print(type)
        p = stat[type]['correct'] / (stat[type]['correct'] + stat[type]['error'] + 0.001)
        # r = stat[type]['correct'] / type_num[type] + 0.001
        r = stat[type]['correct'] / (stat[type]['all'] + 0.001)

        print('Precision: ', p)
        print('Recall: ', r)
        print('F measure: ', (2 * p * r) / (p + r + 0.001))
        print('*'*100)
    other_corr = 0
    other_error = 0
    other_total = 0
    for t in stat.keys():
        if t not in types:
            other_corr += stat[t]['correct']
            other_error += stat[t]['error']
            # other_total += type_num[t]
            other_total += stat[t]['all']
    p = other_corr / (other_corr + other_error)
    r = other_corr / other_total
    print('Other precision: ', p)
    print('Other recall: ', r)
    print('F measure: ', (2 * p * r) / (p + r + 0.001))




def count_unique_errors():
    pred_dir = '/Users/katinska/GramCorr/mtensemble/output/output_exp'
    dir = '/Users/katinska/GramCorr/mtensemble/input/folds_with_spell_1'
    folds_files = [f for f in os.listdir(dir) if f.endswith('csv')]
    files = [el for el in os.listdir(pred_dir) if el.endswith('.pred')]
    errors_input = []
    errors_ens = []
    for f in folds_files:
        with codecs.open(os.path.join(dir, f), mode='r') as table:
            for i, l in enumerate(table):
                if i == 0:
                    continue
                if not l.strip():
                    continue
                data = l.split('\t')
                error = data[0].strip()
                if error not in errors_input:
                    errors_input.append(error)
    for file in files:
        pf = codecs.open(os.path.join(pred_dir, file), 'r', encoding='utf-8')
        lines = pf.readlines()
        for line in lines[1:]:
            if not line.strip():
                continue
            error_ens = line.split(',')[-1].strip()
            if error_ens not in errors_ens:
                errors_ens.append(error_ens)

    print(f'Input errors from all folds: {len(errors_input)}')
    print(f'Errors in pred files: {len(errors_ens)}')
    # Input errors from all folds: 8401
    # Errors in pred files: 8401


def calculate_not_corrected():
    # fold3_2384_ID1233.txt_21_163_163_Hobbies
    dir = '/Users/katinska/GramCorr/mtensemble/input/folds_with_spell_032021_final'
    errors_input = []
    folds_files = [f for f in os.listdir(dir) if f.endswith('csv')]
    for f in folds_files:
        with codecs.open(os.path.join(dir, f), mode='r') as table:
            for i, l in enumerate(table):
                if i == 0:
                    continue
                if not l.strip():
                    continue
                data = l.split('\t')
                error = data[0].strip()
                if error not in errors_input:
                    errors_input.append(error)
    error_coord = '/Users/katinska/GramCorr/data_preprocessing/new_error_coordinates.csv'
    error_c = []
    with codecs.open(error_coord) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for j, row in enumerate(csv_reader):
            if j == 0:
                continue
            error = 'fold'+ row[1] + '_' + row[2] + '_' + row[3] + '_' + row[4] + '_' + row[5] + '_' + row[6] + '_' + row[7]
            # print(error)
            if error not in error_c:
                error_c.append(error)

    out_table = '/Users/katinska/GramCorr/evaluation/out_moses_table_032021.csv'
    err_inp = []
    with codecs.open(out_table) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for j, row in enumerate(csv_reader):
            if j == 0:
                continue
            error = row[0]
            if error not in err_inp:
                err_inp.append(error)
    print(len(errors_input))
    print(len(error_c))
    print(len(err_inp))


# FOLD: fold1
# {"['orth: 02 lcp instead of cap: other cases']": 297, "['orth: 03 cap instead of lcp']": 160, "['orth: 11 omissions: of double consonants']": 153, "['orth: 06 sep instead of tog: compounds']": 112, "['orth: 14 omissions: of one grapheme']": 68, "['orth: 09 tog instead of sep: other cases']": 60, "['orth: 15 insertions: of double consonants']": 55, "['corr: case']": 55, "['orth: 07 sep instead of tog: other cases']": 55, "['orth: 28 Eigennamen']": 54, "['orth: 08 tog instead of sep: minimal phraseologism']": 51, "['orth: 23 transpositions: of consonants: other cases']": 40, "['corr: number']": 40, "['orth: 22 transpositions: of consonants: Fortis/Lenis']": 33, "['orth: 17 insertions: other cases']": 27, "['orth: 21 transpositions: of vowel and umlaut']": 24, "['orth: 04 lcp/cap behind punctuation marks']": 19, "['corr: gender']": 19, "['corr: inflection_paradigm']": 19, "['ommi: incomplete_clause']": 18, "['infl: others']": 17, "['ommi: incomplete_phrase']": 16, "['orth: 27 abbreviations']": 12, "['orth: 06 sep instead of tog: compounds', 'orth: 05 lcp/cap in combination with sep instead of tog/tog instead of sep']": 11, "['orth: 10 missing or unnecessary hyphen']": 10, "['orth: 19 transpositions: of (e/ä) / (ä/e)']": 10, "['orth: 01 lcp instead of cap: polite form']": 10, "['orth: Unknown']": 9, "['corr: unknown']": 9, "['orth: 24 transpositions: of (ss) and (ß)']": 8, "['infl: strong_instead_of_weak']": 8, "['orth: 16 insertions: of a vowel: (ie) instead of (i)']": 8, "['orth: 18 transpositions: of vowels']": 7, "['corr: incorrect_preposition_(po)']": 7, "['woor: deviating']": 6, "['orth: 26 missing or incorrect use of apostrophes']": 6, "['orth: 12 omissions: of a vowel: (i) instead of (ie)']": 5, "['orth: 05 lcp/cap in combination with sep instead of tog/tog instead of sep', 'orth: 06 sep instead of tog: compounds']": 5, "['ommi: ellipses']": 5, "['orth: 13 omissions: of graphemecluster']": 5, "['corr: incorrect_adjunctor']": 4, "['orth: 02 lcp instead of cap: other cases', 'orth: 10 missing or unnecessary hyphen']": 4, "['corr: other']": 4, "['infl: comparision']": 4, "['orth: 10 missing or unnecessary hyphen', 'orth: 02 lcp instead of cap: other cases']": 3, "['orth: 03 cap instead of lcp', 'orth: 17 insertions: other cases']": 2, "['corr: incorrect_preposition_(other_complements)']": 2, "['orth: 15 insertions: of double consonants', 'orth: 03 cap instead of lcp']": 2, "['orth: 23 transpositions: of consonants: other cases', 'orth: 06 sep instead of tog: compounds']": 2, "['woor: v2_instead_of_vl']": 2, "['orth: 24 transpositions: of (ss) and (ß)', 'orth: 14 omissions: of one grapheme']": 1, "['corr: incorrect_pronominal_adverb']": 1, "['orth: 28 Eigennamen', 'orth: 26 missing or incorrect use of apostrophes']": 1, "['orth: 28 Eigennamen', 'orth: 10 missing or unnecessary hyphen', 'orth: 17 insertions: other cases']": 1, "['orth: 11 omissions: of double consonants', 'orth: 15 insertions: of double consonants']": 1, "['orth: 06 sep instead of tog: compounds', 'orth: Unknown']": 1, "['orth: 02 lcp instead of cap: other cases', 'orth: 06 sep instead of tog: compounds']": 1, "['woor: vl_instead_of_v2']": 1, "['orth: 02 lcp instead of cap: other cases', 'orth: 17 insertions: other cases']": 1, "['orth: 11 omissions: of double consonants', 'orth: 07 sep instead of tog: other cases']": 1, "['orth: 14 omissions: of one grapheme', 'orth: 10 missing or unnecessary hyphen']": 1, "['woor: other']": 1, "['orth: 21 transpositions: of vowel and umlaut', 'orth: 06 sep instead of tog: compounds']": 1, "['orth: 11 omissions: of double consonants', 'orth: 18 transpositions: of vowels']": 1, "['orth: Unknown', 'orth: 07 sep instead of tog: other cases']": 1, "['orth: 19 transpositions: of (e/ä) / (ä/e)', 'orth: 03 cap instead of lcp']": 1, "['orth: 20 transpositions: of (i/e) / (e/i)']": 1, "['orth: 09 tog instead of sep: other cases', 'orth: 05 lcp/cap in combination with sep instead of tog/tog instead of sep']": 1, "['orth: 06 sep instead of tog: compounds', 'orth: 05 lcp/cap in combination with sep instead of tog/tog instead of sep', 'orth: 23 transpositions: of consonants: other cases']": 1, "['orth: 09 tog instead of sep: other cases', 'orth: 03 cap instead of lcp']": 1, "['orth: 14 omissions: of one grapheme', 'orth: 03 cap instead of lcp']": 1, "['orth: 23 transpositions: of consonants: other cases', 'orth: 02 lcp instead of cap: other cases']": 1, "['orth: 11 omissions: of double consonants', 'orth: 17 insertions: other cases']": 1, "['orth: 24 transpositions: of (ss) and (ß)', 'orth: 12 omissions: of a vowel: (i) instead of (ie)']": 1, "['orth: 14 omissions: of one grapheme', 'orth: 06 sep instead of tog: compounds']": 1, "['orth: 18 transpositions: of vowels', 'orth: 14 omissions: of one grapheme']": 1, "['orth: 11 omissions: of double consonants', 'orth: 02 lcp instead of cap: other cases']": 1, "['orth: 21 transpositions: of vowel and umlaut', 'orth: 03 cap instead of lcp']": 1, "['orth: 03 cap instead of lcp', 'orth: 15 insertions: of double consonants']": 1, "['orth: 14 omissions: of one grapheme', 'orth: 15 insertions: of double consonants']": 1, "['orth: 14 omissions: of one grapheme', 'orth: 02 lcp instead of cap: other cases']": 1, "['orth: 02 lcp instead of cap: other cases', 'orth: 14 omissions: of one grapheme']": 1, "['orth: 07 sep instead of tog: other cases', 'orth: 09 tog instead of sep: other cases']": 1, "['orth: 09 tog instead of sep: other cases', 'orth: 06 sep instead of tog: compounds']": 1, "['orth: 02 lcp instead of cap: other cases', 'orth: 18 transpositions: of vowels']": 1}
# {'ID1456.txt': 49, 'ID2873.txt': 49, 'ID2518.txt': 49, 'ID2826.txt': 47, 'ID2350.txt': 41, 'ID1730.txt': 34, 'ID2818.txt': 34, 'ID2165.txt': 32, 'ID1535.txt': 30, 'ID1250.txt': 29, 'ID2638.txt': 29, 'ID1120.txt': 29, 'ID2635.txt': 28, 'ID2895.txt': 25, 'ID2558.txt': 25, 'ID2590.txt': 25, 'ID1401.txt': 23, 'ID1241.txt': 23, 'ID2551.txt': 22, 'ID2874.txt': 22, 'ID2958.txt': 22, 'ID2716.txt': 21, 'ID2426.txt': 21, 'ID1043.txt': 20, 'ID2832.txt': 19, 'ID1593.txt': 19, 'ID2351.txt': 18, 'ID2961.txt': 18, 'ID2972.txt': 17, 'ID1092.txt': 16, 'ID1109.txt': 16, 'ID2170.txt': 15, 'ID2932.txt': 15, 'ID2682.txt': 15, 'ID1381.txt': 15, 'ID2187.txt': 14, 'ID1175.txt': 14, 'ID1451.txt': 13, 'ID1732.txt': 13, 'ID1846.txt': 12, 'ID1719.txt': 12, 'ID2364.txt': 11, 'ID1476.txt': 11, 'ID2129.txt': 11, 'ID1753.txt': 11, 'ID1248.txt': 11, 'ID2714.txt': 11, 'ID2139.txt': 11, 'ID1683.txt': 11, 'ID1143.txt': 11, 'ID1842.txt': 10, 'ID1326.txt': 10, 'ID2133.txt': 10, 'ID1633.txt': 10, 'ID1994.txt': 10, 'ID2894.txt': 10, 'ID2529.txt': 9, 'ID2539.txt': 9, 'ID1180.txt': 9, 'ID2349.txt': 9, 'ID1520.txt': 9, 'ID2791.txt': 9, 'ID2301.txt': 9, 'ID2420.txt': 9, 'ID1510.txt': 8, 'ID2688.txt': 8, 'ID1129.txt': 8, 'ID1517.txt': 8, 'ID1378.txt': 8, 'ID1494.txt': 8, 'ID2257.txt': 8, 'ID2969.txt': 7, 'ID2983.txt': 7, 'ID1712.txt': 7, 'ID1672.txt': 7, 'ID1512.txt': 7, 'ID2893.txt': 7, 'ID2903.txt': 7, 'ID2929.txt': 7, 'ID2425.txt': 7, 'ID2978.txt': 7, 'ID1670.txt': 7, 'ID2926.txt': 7, 'ID1505.txt': 6, 'ID2653.txt': 6, 'ID1689.txt': 6, 'ID1937.txt': 6, 'ID1806.txt': 6, 'ID2395.txt': 6, 'ID2636.txt': 6, 'ID1739.txt': 6, 'ID1131.txt': 6, 'ID2504.txt': 6, 'ID2219.txt': 6, 'ID2699.txt': 6, 'ID1276.txt': 6, 'ID1894.txt': 6, 'ID2735.txt': 5, 'ID1516.txt': 5, 'ID2479.txt': 5, 'ID1649.txt': 5, 'ID1178.txt': 5, 'ID2614.txt': 5, 'ID2630.txt': 5, 'ID2123.txt': 5, 'ID2256.txt': 5, 'ID1569.txt': 5, 'ID1865.txt': 5, 'ID2331.txt': 4, 'ID2029.txt': 4, 'ID1941.txt': 4, 'ID1892.txt': 4, 'ID1693.txt': 4, 'ID1275.txt': 4, 'ID2432.txt': 4, 'ID2198.txt': 4, 'ID2334.txt': 4, 'ID2039.txt': 4, 'ID2326.txt': 3, 'ID1756.txt': 3, 'ID1856.txt': 3, 'ID2476.txt': 3, 'ID1785.txt': 3, 'ID2002.txt': 3, 'ID2737.txt': 3, 'ID2336.txt': 3, 'ID2955.txt': 3, 'ID1965.txt': 3, 'ID1775.txt': 3, 'ID2339.txt': 2, 'ID2568.txt': 2, 'ID2199.txt': 2, 'ID1926.txt': 2, 'ID2034.txt': 2, 'ID2136.txt': 2, 'ID2346.txt': 2, 'ID2742.txt': 2, 'ID1245.txt': 1, 'ID1835.txt': 1, 'ID2615.txt': 1, 'ID2562.txt': 1, 'ID2036.txt': 1, 'ID2632.txt': 1, 'ID2780.txt': 1, 'ID1820.txt': 1}
# FOLD: fold2
# {"['orth: 02 lcp instead of cap: other cases']": 196, "['orth: 03 cap instead of lcp']": 171, "['orth: 11 omissions: of double consonants']": 147, "['orth: 06 sep instead of tog: compounds']": 112, "['orth: 07 sep instead of tog: other cases']": 72, "['orth: 14 omissions: of one grapheme']": 71, "['orth: 28 Eigennamen']": 60, "['corr: case']": 58, "['orth: 15 insertions: of double consonants']": 51, "['orth: 09 tog instead of sep: other cases']": 46, "['orth: 08 tog instead of sep: minimal phraseologism']": 45, "['orth: 23 transpositions: of consonants: other cases']": 37, "['orth: 17 insertions: other cases']": 33, "['corr: number']": 32, "['orth: 22 transpositions: of consonants: Fortis/Lenis']": 25, "['corr: gender']": 25, "['orth: 21 transpositions: of vowel and umlaut']": 18, "['orth: 18 transpositions: of vowels']": 17, "['corr: inflection_paradigm']": 15, "['orth: 10 missing or unnecessary hyphen']": 14, "['orth: 27 abbreviations']": 13, "['orth: 04 lcp/cap behind punctuation marks']": 13, "['ommi: incomplete_phrase']": 12, "['infl: others']": 12, "['orth: 16 insertions: of a vowel: (ie) instead of (i)']": 11, "['orth: 26 missing or incorrect use of apostrophes']": 10, "['ommi: incomplete_clause']": 9, "['orth: 12 omissions: of a vowel: (i) instead of (ie)']": 9, "['orth: 05 lcp/cap in combination with sep instead of tog/tog instead of sep', 'orth: 06 sep instead of tog: compounds']": 9, "['orth: 13 omissions: of graphemecluster']": 7, "['corr: incorrect_preposition_(po)']": 7, "['infl: comparision']": 7, "['orth: 01 lcp instead of cap: polite form']": 7, "['orth: 19 transpositions: of (e/ä) / (ä/e)']": 6, "['orth: 24 transpositions: of (ss) and (ß)']": 5, "['corr: unknown']": 5, "['orth: Unknown']": 4, "['orth: 25 incorrect positioning of two graphemes']": 4, "['woor: deviating']": 3, "['infl: strong_instead_of_weak']": 3, "['orth: 21 transpositions: of vowel and umlaut', 'orth: 14 omissions: of one grapheme']": 3, "['corr: incorrect_pronominal_adverb']": 2, "['woor: other']": 2, "['infl: weak_instead_of_strong']": 2, "['corr: incorrect_preposition_(other_complements)']": 2, "['orth: 11 omissions: of double consonants', 'orth: 15 insertions: of double consonants']": 2, "['orth: 06 sep instead of tog: compounds', 'orth: 05 lcp/cap in combination with sep instead of tog/tog instead of sep']": 2, "['orth: 20 transpositions: of (i/e) / (e/i)']": 2, "['ommi: ellipses']": 2, "['orth: 19 transpositions: of (e/ä) / (ä/e)', 'orth: 03 cap instead of lcp']": 2, "['orth: 06 sep instead of tog: compounds', 'orth: 11 omissions: of double consonants']": 2, "['orth: 14 omissions: of one grapheme', 'orth: 02 lcp instead of cap: other cases']": 2, "['orth: 03 cap instead of lcp', 'orth: 14 omissions: of one grapheme']": 2, "['orth: 19 transpositions: of (e/ä) / (ä/e)', 'orth: 22 transpositions: of consonants: Fortis/Lenis']": 1, "['orth: 03 cap instead of lcp', 'orth: 10 missing or unnecessary hyphen']": 1, "['orth: 21 transpositions: of vowel and umlaut', 'orth: 07 sep instead of tog: other cases']": 1, "['orth: 07 sep instead of tog: other cases', 'orth: 27 abbreviations']": 1, "['orth: 27 abbreviations', 'orth: 18 transpositions: of vowels']": 1, "['woor: vl_instead_of_v2']": 1, "['woor: v2_instead_of_v1']": 1, "['orth: 17 insertions: other cases', 'orth: 14 omissions: of one grapheme']": 1, "['orth: 03 cap instead of lcp', 'orth: 06 sep instead of tog: compounds']": 1, "['orth: 20 transpositions: of (i/e) / (e/i)', 'orth: 22 transpositions: of consonants: Fortis/Lenis']": 1, "['woor: v2_instead_of_vl']": 1, "['orth: 02 lcp instead of cap: other cases', 'orth: 11 omissions: of double consonants']": 1, "['orth: 11 omissions: of double consonants', 'orth: 19 transpositions: of (e/ä) / (ä/e)']": 1, "['orth: 23 transpositions: of consonants: other cases', 'orth: 11 omissions: of double consonants']": 1, "['orth: 02 lcp instead of cap: other cases', 'orth: 10 missing or unnecessary hyphen']": 1, "['orth: 14 omissions: of one grapheme', 'orth: 21 transpositions: of vowel and umlaut']": 1, "['orth: 06 sep instead of tog: compounds', 'orth: 23 transpositions: of consonants: other cases']": 1, "['orth: 22 transpositions: of consonants: Fortis/Lenis', 'orth: 17 insertions: other cases']": 1, "['orth: 11 omissions: of double consonants', 'orth: 16 insertions: of a vowel: (ie) instead of (i)']": 1, "['orth: 17 insertions: other cases', 'orth: 11 omissions: of double consonants']": 1, "['orth: 15 insertions: of double consonants', 'orth: 11 omissions: of double consonants']": 1, "['orth: 07 sep instead of tog: other cases', 'orth: 02 lcp instead of cap: other cases']": 1, "['corr: incorrect_adjunctor']": 1, "['orth: 12 omissions: of a vowel: (i) instead of (ie)', 'orth: 15 insertions: of double consonants']": 1}
# {'ID2138.txt': 41, 'ID1095.txt': 39, 'ID1163.txt': 32, 'ID2362.txt': 30, 'ID2287.txt': 30, 'ID2513.txt': 29, 'ID1016.txt': 26, 'ID1539.txt': 24, 'ID1975.txt': 24, 'ID1183.txt': 23, 'ID2276.txt': 23, 'ID1398.txt': 23, 'ID1472.txt': 22, 'ID2464.txt': 22, 'ID1875.txt': 20, 'ID2205.txt': 18, 'ID1081.txt': 18, 'ID1630.txt': 18, 'ID2360.txt': 17, 'ID2912.txt': 17, 'ID2312.txt': 17, 'ID1123.txt': 17, 'ID1847.txt': 17, 'ID1045.txt': 16, 'ID1546.txt': 16, 'ID1715.txt': 16, 'ID2945.txt': 15, 'ID1460.txt': 15, 'ID1012.txt': 14, 'ID2332.txt': 14, 'ID2403.txt': 13, 'ID1313.txt': 13, 'ID2130.txt': 13, 'ID2023.txt': 12, 'ID1974.txt': 12, 'ID2651.txt': 12, 'ID2200.txt': 12, 'ID2221.txt': 12, 'ID1788.txt': 12, 'ID1599.txt': 11, 'ID2444.txt': 11, 'ID1496.txt': 11, 'ID1641.txt': 11, 'ID1750.txt': 11, 'ID1144.txt': 11, 'ID1302.txt': 11, 'ID1483.txt': 10, 'ID1130.txt': 10, 'ID2889.txt': 10, 'ID1954.txt': 10, 'ID2052.txt': 10, 'ID1923.txt': 10, 'ID2240.txt': 10, 'ID1209.txt': 10, 'ID1957.txt': 10, 'ID1825.txt': 10, 'ID2266.txt': 10, 'ID1048.txt': 10, 'ID1104.txt': 10, 'ID2613.txt': 9, 'ID1605.txt': 9, 'ID1265.txt': 9, 'ID1013.txt': 9, 'ID2069.txt': 9, 'ID2559.txt': 9, 'ID2897.txt': 9, 'ID1686.txt': 8, 'ID1811.txt': 8, 'ID2525.txt': 8, 'ID1549.txt': 8, 'ID2038.txt': 8, 'ID2260.txt': 8, 'ID1047.txt': 8, 'ID2313.txt': 8, 'ID2671.txt': 8, 'ID1060.txt': 8, 'ID1924.txt': 8, 'ID2566.txt': 8, 'ID1523.txt': 8, 'ID1949.txt': 8, 'ID2004.txt': 8, 'ID2186.txt': 8, 'ID1145.txt': 7, 'ID1833.txt': 7, 'ID1486.txt': 7, 'ID1780.txt': 7, 'ID2594.txt': 7, 'ID1407.txt': 7, 'ID1458.txt': 7, 'ID1981.txt': 7, 'ID2668.txt': 7, 'ID1069.txt': 7, 'ID1421.txt': 7, 'ID1762.txt': 7, 'ID1208.txt': 6, 'ID1874.txt': 6, 'ID1964.txt': 6, 'ID1447.txt': 6, 'ID1466.txt': 6, 'ID2118.txt': 6, 'ID2168.txt': 6, 'ID2799.txt': 6, 'ID2743.txt': 6, 'ID2778.txt': 5, 'ID1235.txt': 5, 'ID1484.txt': 5, 'ID2228.txt': 5, 'ID1426.txt': 5, 'ID2320.txt': 5, 'ID1508.txt': 5, 'ID1500.txt': 5, 'ID1839.txt': 5, 'ID2792.txt': 4, 'ID1160.txt': 4, 'ID1878.txt': 4, 'ID2916.txt': 4, 'ID1726.txt': 4, 'ID1677.txt': 4, 'ID2049.txt': 4, 'ID1372.txt': 4, 'ID2172.txt': 4, 'ID2538.txt': 4, 'ID2182.txt': 4, 'ID2439.txt': 4, 'ID2300.txt': 3, 'ID2765.txt': 3, 'ID1032.txt': 3, 'ID1236.txt': 3, 'ID1645.txt': 3, 'ID2480.txt': 3, 'ID2763.txt': 3, 'ID2295.txt': 3, 'ID1616.txt': 3, 'ID2322.txt': 3, 'ID1452.txt': 2, 'ID2046.txt': 2, 'ID2675.txt': 2, 'ID2393.txt': 2, 'ID2759.txt': 2, 'ID2265.txt': 2, 'ID2626.txt': 2, 'ID1226.txt': 2, 'ID1138.txt': 2, 'ID2014.txt': 2, 'ID2121.txt': 2, 'ID2621.txt': 2, 'ID1299.txt': 2, 'ID2262.txt': 1, 'ID1674.txt': 1, 'ID2600.txt': 1}
# FOLD: fold3
# {"['orth: 02 lcp instead of cap: other cases']": 244, "['orth: 03 cap instead of lcp']": 183, "['orth: 11 omissions: of double consonants']": 154, "['orth: 06 sep instead of tog: compounds']": 125, "['orth: 14 omissions: of one grapheme']": 83, "['orth: 07 sep instead of tog: other cases']": 68, "['corr: case']": 68, "['orth: 28 Eigennamen']": 65, "['orth: 17 insertions: other cases']": 47, "['orth: 23 transpositions: of consonants: other cases']": 46, "['orth: 15 insertions: of double consonants']": 42, "['orth: 08 tog instead of sep: minimal phraseologism']": 38, "['orth: 09 tog instead of sep: other cases']": 38, "['orth: 04 lcp/cap behind punctuation marks']": 35, "['corr: number']": 25, "['orth: 22 transpositions: of consonants: Fortis/Lenis']": 23, "['orth: 21 transpositions: of vowel and umlaut']": 21, "['orth: 10 missing or unnecessary hyphen']": 19, "['corr: inflection_paradigm']": 17, "['corr: gender']": 17, "['orth: 18 transpositions: of vowels']": 14, "['ommi: incomplete_clause']": 13, "['orth: 24 transpositions: of (ss) and (ß)']": 13, "['orth: 01 lcp instead of cap: polite form']": 13, "['orth: 27 abbreviations']": 13, "['infl: others']": 12, "['corr: incorrect_preposition_(po)']": 12, "['orth: 16 insertions: of a vowel: (ie) instead of (i)']": 11, "['orth: 12 omissions: of a vowel: (i) instead of (ie)']": 10, "['ommi: ellipses']": 10, "['ommi: incomplete_phrase']": 9, "['orth: 13 omissions: of graphemecluster']": 9, "['corr: unknown']": 9, "['infl: comparision']": 9, "['orth: 19 transpositions: of (e/ä) / (ä/e)']": 8, "['orth: Unknown']": 8, "['orth: 26 missing or incorrect use of apostrophes']": 7, "['orth: 06 sep instead of tog: compounds', 'orth: 05 lcp/cap in combination with sep instead of tog/tog instead of sep']": 5, "['infl: strong_instead_of_weak']": 4, "['corr: other']": 3, "['corr: incorrect_preposition_(other_complements)']": 3, "['corr: incorrect_adjunctor']": 2, "['orth: 17 insertions: other cases', 'orth: 14 omissions: of one grapheme']": 2, "['corr: person']": 2, "['corr: incorrect_pronominal_adverb']": 2, "['orth: 23 transpositions: of consonants: other cases', 'orth: 06 sep instead of tog: compounds']": 1, "['orth: 23 transpositions: of consonants: other cases', 'orth: 19 transpositions: of (e/ä) / (ä/e)']": 1, "['orth: 02 lcp instead of cap: other cases', 'orth: 10 missing or unnecessary hyphen']": 1, "['orth: 09 tog instead of sep: other cases', 'orth: 05 lcp/cap in combination with sep instead of tog/tog instead of sep']": 1, "['orth: 15 insertions: of double consonants', 'orth: 06 sep instead of tog: compounds']": 1, "['orth: 02 lcp instead of cap: other cases', 'orth: 23 transpositions: of consonants: other cases']": 1, "['orth: 02 lcp instead of cap: other cases', 'orth: 14 omissions: of one grapheme']": 1, "['orth: 03 cap instead of lcp', 'orth: 14 omissions: of one grapheme']": 1, "['orth: Unknown', 'orth: 02 lcp instead of cap: other cases']": 1, "['woor: v2_instead_of_vl']": 1, "['orth: 05 lcp/cap in combination with sep instead of tog/tog instead of sep', 'orth: 09 tog instead of sep: other cases']": 1, "['orth: 10 missing or unnecessary hyphen', 'orth: 02 lcp instead of cap: other cases']": 1, "['orth: 10 missing or unnecessary hyphen', 'orth: 25 incorrect positioning of two graphemes']": 1, "['woor: v1_instead_of_v2']": 1, "['orth: 14 omissions: of one grapheme', 'orth: 23 transpositions: of consonants: other cases']": 1, "['orth: 06 sep instead of tog: compounds', 'orth: 17 insertions: other cases']": 1, "['orth: 06 sep instead of tog: compounds', 'orth: 14 omissions: of one grapheme', 'orth: 05 lcp/cap in combination with sep instead of tog/tog instead of sep']": 1, "['orth: 19 transpositions: of (e/ä) / (ä/e)', 'orth: 14 omissions: of one grapheme']": 1, "['orth: 14 omissions: of one grapheme', 'orth: 02 lcp instead of cap: other cases']": 1, "['woor: deviating']": 1, "['orth: 10 missing or unnecessary hyphen', 'orth: 04 lcp/cap behind punctuation marks']": 1, "['orth: 03 cap instead of lcp', 'orth: 11 omissions: of double consonants']": 1, "['orth: 13 omissions: of graphemecluster', 'orth: 03 cap instead of lcp']": 1, "['orth: 03 cap instead of lcp', 'orth: 22 transpositions: of consonants: Fortis/Lenis']": 1, "['orth: 05 lcp/cap in combination with sep instead of tog/tog instead of sep', 'orth: 06 sep instead of tog: compounds']": 1, "['orth: 17 insertions: other cases', 'orth: 19 transpositions: of (e/ä) / (ä/e)']": 1, "['orth: 07 sep instead of tog: other cases', 'orth: 02 lcp instead of cap: other cases']": 1, "['orth: 10 missing or unnecessary hyphen', 'orth: 03 cap instead of lcp']": 1, "['orth: 23 transpositions: of consonants: other cases', 'orth: 09 tog instead of sep: other cases']": 1, "['orth: 14 omissions: of one grapheme', 'orth: 17 insertions: other cases']": 1, "['woor: vl_instead_of_v2']": 1, "['orth: 02 lcp instead of cap: other cases', 'orth: 11 omissions: of double consonants']": 1, "['orth: 22 transpositions: of consonants: Fortis/Lenis', 'orth: 14 omissions: of one grapheme']": 1}
# {'ID1734.txt': 59, 'ID2283.txt': 36, 'ID1386.txt': 36, 'ID1547.txt': 35, 'ID2819.txt': 35, 'ID1902.txt': 31, 'ID1380.txt': 29, 'ID2549.txt': 26, 'ID1082.txt': 26, 'ID1962.txt': 24, 'ID2213.txt': 24, 'ID1474.txt': 24, 'ID2634.txt': 23, 'ID2779.txt': 23, 'ID2808.txt': 21, 'ID1352.txt': 21, 'ID1307.txt': 21, 'ID2293.txt': 21, 'ID2128.txt': 20, 'ID2760.txt': 19, 'ID1283.txt': 19, 'ID2438.txt': 19, 'ID1721.txt': 19, 'ID2740.txt': 19, 'ID2225.txt': 18, 'ID2448.txt': 18, 'ID1223.txt': 18, 'ID2218.txt': 18, 'ID2927.txt': 17, 'ID1093.txt': 17, 'ID1493.txt': 17, 'ID2215.txt': 16, 'ID1394.txt': 16, 'ID1147.txt': 16, 'ID1489.txt': 15, 'ID2212.txt': 15, 'ID1728.txt': 15, 'ID2542.txt': 15, 'ID2290.txt': 14, 'ID1761.txt': 14, 'ID2753.txt': 13, 'ID2000.txt': 13, 'ID1230.txt': 13, 'ID2957.txt': 12, 'ID2864.txt': 12, 'ID2100.txt': 12, 'ID2183.txt': 12, 'ID2655.txt': 12, 'ID1116.txt': 11, 'ID2321.txt': 11, 'ID2813.txt': 11, 'ID1290.txt': 11, 'ID1617.txt': 11, 'ID2865.txt': 11, 'ID2180.txt': 11, 'ID2809.txt': 11, 'ID2191.txt': 11, 'ID2872.txt': 11, 'ID2586.txt': 11, 'ID2484.txt': 11, 'ID1159.txt': 10, 'ID2282.txt': 10, 'ID1266.txt': 10, 'ID2325.txt': 10, 'ID1798.txt': 9, 'ID1121.txt': 9, 'ID2823.txt': 9, 'ID1545.txt': 9, 'ID2944.txt': 9, 'ID1309.txt': 9, 'ID1492.txt': 9, 'ID1084.txt': 9, 'ID2904.txt': 8, 'ID1037.txt': 8, 'ID1204.txt': 8, 'ID2664.txt': 8, 'ID2729.txt': 8, 'ID2801.txt': 8, 'ID2292.txt': 8, 'ID1194.txt': 8, 'ID1643.txt': 8, 'ID1973.txt': 7, 'ID2093.txt': 7, 'ID2733.txt': 7, 'ID2379.txt': 7, 'ID2071.txt': 7, 'ID2696.txt': 7, 'ID2335.txt': 7, 'ID2616.txt': 7, 'ID2769.txt': 7, 'ID1468.txt': 7, 'ID2342.txt': 6, 'ID1582.txt': 6, 'ID1826.txt': 6, 'ID2988.txt': 6, 'ID2868.txt': 6, 'ID2843.txt': 6, 'ID2678.txt': 6, 'ID2798.txt': 6, 'ID1340.txt': 6, 'ID1908.txt': 6, 'ID2796.txt': 5, 'ID1357.txt': 5, 'ID2190.txt': 5, 'ID2413.txt': 5, 'ID1388.txt': 5, 'ID1998.txt': 5, 'ID2905.txt': 5, 'ID2166.txt': 5, 'ID1555.txt': 5, 'ID1228.txt': 5, 'ID1128.txt': 5, 'ID2010.txt': 5, 'ID2443.txt': 4, 'ID1252.txt': 4, 'ID2318.txt': 4, 'ID1049.txt': 4, 'ID2531.txt': 4, 'ID1479.txt': 4, 'ID1522.txt': 4, 'ID2264.txt': 4, 'ID2578.txt': 4, 'ID1967.txt': 4, 'ID1217.txt': 3, 'ID2609.txt': 3, 'ID1482.txt': 3, 'ID2041.txt': 3, 'ID1687.txt': 3, 'ID2523.txt': 3, 'ID1861.txt': 3, 'ID2236.txt': 3, 'ID1046.txt': 3, 'ID2304.txt': 3, 'ID1554.txt': 3, 'ID2348.txt': 3, 'ID1579.txt': 2, 'ID2461.txt': 2, 'ID1970.txt': 2, 'ID2067.txt': 2, 'ID1667.txt': 2, 'ID1606.txt': 2, 'ID1504.txt': 2, 'ID2311.txt': 2, 'ID1297.txt': 2, 'ID2382.txt': 2, 'ID2045.txt': 1, 'ID1790.txt': 1, 'ID1030.txt': 1, 'ID1199.txt': 1}
# FOLD: fold4
# {"['orth: 02 lcp instead of cap: other cases']": 231, "['orth: 03 cap instead of lcp']": 142, "['orth: 11 omissions: of double consonants']": 119, "['orth: 06 sep instead of tog: compounds']": 99, "['orth: 08 tog instead of sep: minimal phraseologism']": 59, "['orth: 28 Eigennamen']": 57, "['orth: 14 omissions: of one grapheme']": 53, "['corr: case']": 51, "['orth: 07 sep instead of tog: other cases']": 49, "['orth: 23 transpositions: of consonants: other cases']": 42, "['orth: 15 insertions: of double consonants']": 38, "['orth: 09 tog instead of sep: other cases']": 32, "['orth: 22 transpositions: of consonants: Fortis/Lenis']": 28, "['corr: gender']": 23, "['corr: number']": 22, "['orth: 17 insertions: other cases']": 22, "['orth: 04 lcp/cap behind punctuation marks']": 21, "['orth: 21 transpositions: of vowel and umlaut']": 21, "['orth: 19 transpositions: of (e/ä) / (ä/e)']": 13, "['orth: 10 missing or unnecessary hyphen']": 13, "['infl: others']": 12, "['ommi: incomplete_clause']": 12, "['orth: 18 transpositions: of vowels']": 10, "['corr: inflection_paradigm']": 10, "['orth: 16 insertions: of a vowel: (ie) instead of (i)']": 10, "['orth: 27 abbreviations']": 10, "['orth: 13 omissions: of graphemecluster']": 9, "['ommi: incomplete_phrase']": 9, "['ommi: ellipses']": 8, "['orth: 06 sep instead of tog: compounds', 'orth: 05 lcp/cap in combination with sep instead of tog/tog instead of sep']": 6, "['orth: Unknown']": 6, "['orth: 26 missing or incorrect use of apostrophes']": 6, "['corr: unknown']": 5, "['orth: 12 omissions: of a vowel: (i) instead of (ie)']": 5, "['corr: incorrect_preposition_(po)']": 5, "['orth: 01 lcp instead of cap: polite form']": 5, "['orth: 05 lcp/cap in combination with sep instead of tog/tog instead of sep', 'orth: 06 sep instead of tog: compounds']": 4, "['orth: 24 transpositions: of (ss) and (ß)']": 4, "['infl: strong_instead_of_weak']": 3, "['woor: deviating']": 3, "['orth: 25 incorrect positioning of two graphemes']": 3, "['orth: 20 transpositions: of (i/e) / (e/i)']": 3, "['corr: incorrect_preposition_(other_complements)']": 2, "['woor: other']": 2, "['orth: 03 cap instead of lcp', 'orth: 11 omissions: of double consonants']": 2, "['orth: 02 lcp instead of cap: other cases', 'orth: 14 omissions: of one grapheme']": 2, "['woor: vl_instead_of_v2']": 2, "['corr: other']": 2, "['orth: 06 sep instead of tog: compounds', 'orth: 09 tog instead of sep: other cases']": 2, "['orth: 10 missing or unnecessary hyphen', 'orth: 02 lcp instead of cap: other cases']": 2, "['orth: 20 transpositions: of (i/e) / (e/i)', 'orth: 23 transpositions: of consonants: other cases']": 2, "['orth: 10 missing or unnecessary hyphen', 'orth: 07 sep instead of tog: other cases']": 1, "['orth: 07 sep instead of tog: other cases', 'orth: 02 lcp instead of cap: other cases']": 1, "['orth: 22 transpositions: of consonants: Fortis/Lenis', 'orth: 14 omissions: of one grapheme']": 1, "['orth: 17 insertions: other cases', 'orth: 07 sep instead of tog: other cases']": 1, "['orth: 25 incorrect positioning of two graphemes', 'orth: 21 transpositions: of vowel and umlaut']": 1, "['orth: 11 omissions: of double consonants', 'orth: 25 incorrect positioning of two graphemes']": 1, "['orth: 22 transpositions: of consonants: Fortis/Lenis', 'orth: Unknown']": 1, "['orth: 06 sep instead of tog: compounds', 'orth: 02 lcp instead of cap: other cases']": 1, "['orth: 14 omissions: of one grapheme', 'orth: 02 lcp instead of cap: other cases']": 1, "['orth: 22 transpositions: of consonants: Fortis/Lenis', 'orth: 02 lcp instead of cap: other cases']": 1, "['orth: 17 insertions: other cases', 'orth: 13 omissions: of graphemecluster']": 1, "['orth: 14 omissions: of one grapheme', 'orth: 06 sep instead of tog: compounds']": 1, "['infl: comparision']": 1, "['corr: ambiguous_pronominal_adverb']": 1, "['orth: 04 lcp/cap behind punctuation marks', 'orth: 07 sep instead of tog: other cases']": 1, "['orth: 17 insertions: other cases', 'orth: 02 lcp instead of cap: other cases', 'orth: 10 missing or unnecessary hyphen']": 1, "['corr: incorrect_pronominal_adverb']": 1, "['orth: 17 insertions: other cases', 'orth: 09 tog instead of sep: other cases']": 1, "['woor: v1_instead_of_v2']": 1, "['orth: 06 sep instead of tog: compounds', 'orth: 17 insertions: other cases']": 1, "['orth: 17 insertions: other cases', 'orth: 11 omissions: of double consonants', 'orth: 14 omissions: of one grapheme']": 1, "['orth: 14 omissions: of one grapheme', 'orth: 17 insertions: other cases']": 1, "['orth: 16 insertions: of a vowel: (ie) instead of (i)', 'orth: 15 insertions: of double consonants']": 1, "['orth: 06 sep instead of tog: compounds', 'orth: 10 missing or unnecessary hyphen']": 1, "['orth: 24 transpositions: of (ss) and (ß)', 'orth: 22 transpositions: of consonants: Fortis/Lenis']": 1, "['orth: 02 lcp instead of cap: other cases', 'orth: 10 missing or unnecessary hyphen']": 1, "['woor: v2_instead_of_vl']": 1, "['orth: 05 lcp/cap in combination with sep instead of tog/tog instead of sep', 'orth: 09 tog instead of sep: other cases']": 1, "['orth: 11 omissions: of double consonants', 'orth: 17 insertions: other cases']": 1, "['orth: 02 lcp instead of cap: other cases', 'orth: 09 tog instead of sep: other cases', 'orth: 23 transpositions: of consonants: other cases']": 1, "['corr: incorrect_adjunctor']": 1, "['orth: 17 insertions: other cases', 'orth: 11 omissions: of double consonants']": 1}
# {'ID2192.txt': 55, 'ID1091.txt': 41, 'ID2519.txt': 39, 'ID1922.txt': 28, 'ID2208.txt': 28, 'ID1928.txt': 27, 'ID2840.txt': 25, 'ID2153.txt': 25, 'ID2328.txt': 23, 'ID2164.txt': 22, 'ID1162.txt': 20, 'ID1534.txt': 19, 'ID2207.txt': 18, 'ID2005.txt': 18, 'ID1379.txt': 18, 'ID2968.txt': 18, 'ID1989.txt': 17, 'ID1303.txt': 17, 'ID2974.txt': 17, 'ID2224.txt': 17, 'ID2126.txt': 16, 'ID1863.txt': 16, 'ID2363.txt': 16, 'ID2428.txt': 15, 'ID1669.txt': 14, 'ID2107.txt': 14, 'ID1100.txt': 14, 'ID2054.txt': 14, 'ID2687.txt': 14, 'ID2369.txt': 13, 'ID2837.txt': 13, 'ID2625.txt': 13, 'ID1355.txt': 13, 'ID2629.txt': 13, 'ID1404.txt': 13, 'ID1155.txt': 13, 'ID1757.txt': 13, 'ID1260.txt': 12, 'ID1086.txt': 11, 'ID2593.txt': 11, 'ID1052.txt': 11, 'ID2374.txt': 11, 'ID2468.txt': 11, 'ID1044.txt': 11, 'ID2784.txt': 11, 'ID1231.txt': 11, 'ID1763.txt': 11, 'ID1420.txt': 10, 'ID1685.txt': 10, 'ID1390.txt': 10, 'ID2612.txt': 9, 'ID1988.txt': 9, 'ID1907.txt': 9, 'ID1243.txt': 9, 'ID1382.txt': 9, 'ID2053.txt': 9, 'ID2707.txt': 9, 'ID2715.txt': 8, 'ID1061.txt': 8, 'ID1263.txt': 8, 'ID1444.txt': 8, 'ID1743.txt': 8, 'ID1406.txt': 8, 'ID1211.txt': 8, 'ID1548.txt': 7, 'ID2414.txt': 7, 'ID2195.txt': 7, 'ID1987.txt': 7, 'ID2920.txt': 7, 'ID2681.txt': 7, 'ID2694.txt': 7, 'ID2323.txt': 7, 'ID1935.txt': 7, 'ID1903.txt': 7, 'ID1464.txt': 6, 'ID2649.txt': 6, 'ID1958.txt': 6, 'ID1956.txt': 6, 'ID2913.txt': 6, 'ID1631.txt': 6, 'ID1197.txt': 6, 'ID2919.txt': 6, 'ID2344.txt': 6, 'ID1625.txt': 6, 'ID1868.txt': 6, 'ID1740.txt': 6, 'ID2830.txt': 5, 'ID1344.txt': 5, 'ID2062.txt': 5, 'ID2852.txt': 5, 'ID1023.txt': 5, 'ID2935.txt': 5, 'ID2732.txt': 5, 'ID2063.txt': 5, 'ID1233.txt': 5, 'ID1333.txt': 5, 'ID1301.txt': 5, 'ID1294.txt': 5, 'ID1553.txt': 5, 'ID1206.txt': 5, 'ID1784.txt': 5, 'ID2844.txt': 4, 'ID2762.txt': 4, 'ID2244.txt': 4, 'ID1501.txt': 4, 'ID1430.txt': 4, 'ID2249.txt': 4, 'ID2040.txt': 4, 'ID1580.txt': 4, 'ID2980.txt': 4, 'ID1795.txt': 4, 'ID1877.txt': 4, 'ID1699.txt': 4, 'ID2854.txt': 4, 'ID2552.txt': 3, 'ID2686.txt': 3, 'ID1867.txt': 3, 'ID1083.txt': 3, 'ID2930.txt': 3, 'ID2436.txt': 3, 'ID1139.txt': 3, 'ID1446.txt': 3, 'ID1802.txt': 3, 'ID2812.txt': 3, 'ID2009.txt': 3, 'ID1587.txt': 3, 'ID1791.txt': 3, 'ID1759.txt': 3, 'ID1096.txt': 3, 'ID2563.txt': 3, 'ID1577.txt': 3, 'ID1328.txt': 3, 'ID1202.txt': 3, 'ID1888.txt': 2, 'ID1216.txt': 2, 'ID1782.txt': 2, 'ID1004.txt': 2, 'ID2522.txt': 2, 'ID2744.txt': 1, 'ID1692.txt': 1, 'ID2341.txt': 1, 'ID1221.txt': 1, 'ID2179.txt': 1, 'ID2620.txt': 1, 'ID1132.txt': 1, 'ID2402.txt': 1}
# FOLD: fold5
# {"['orth: 02 lcp instead of cap: other cases']": 207, "['orth: 03 cap instead of lcp']": 184, "['orth: 11 omissions: of double consonants']": 130, "['orth: 06 sep instead of tog: compounds']": 125, "['orth: 08 tog instead of sep: minimal phraseologism']": 61, "['corr: case']": 55, "['orth: 07 sep instead of tog: other cases']": 54, "['orth: 14 omissions: of one grapheme']": 49, "['orth: 17 insertions: other cases']": 42, "['corr: number']": 41, "['orth: 15 insertions: of double consonants']": 37, "['orth: 28 Eigennamen']": 34, "['orth: 09 tog instead of sep: other cases']": 33, "['orth: 23 transpositions: of consonants: other cases']": 32, "['corr: gender']": 22, "['orth: 27 abbreviations']": 21, "['orth: 10 missing or unnecessary hyphen']": 21, "['orth: 04 lcp/cap behind punctuation marks']": 20, "['orth: 22 transpositions: of consonants: Fortis/Lenis']": 19, "['orth: 21 transpositions: of vowel and umlaut']": 19, "['ommi: incomplete_clause']": 14, "['orth: 06 sep instead of tog: compounds', 'orth: 05 lcp/cap in combination with sep instead of tog/tog instead of sep']": 13, "['corr: incorrect_preposition_(po)']": 12, "['ommi: incomplete_phrase']": 12, "['orth: 16 insertions: of a vowel: (ie) instead of (i)']": 12, "['orth: 24 transpositions: of (ss) and (ß)']": 11, "['orth: 19 transpositions: of (e/ä) / (ä/e)']": 9, "['orth: 13 omissions: of graphemecluster']": 9, "['corr: inflection_paradigm']": 8, "['orth: 18 transpositions: of vowels']": 8, "['infl: strong_instead_of_weak']": 8, "['orth: 05 lcp/cap in combination with sep instead of tog/tog instead of sep', 'orth: 06 sep instead of tog: compounds']": 7, "['orth: 12 omissions: of a vowel: (i) instead of (ie)']": 7, "['ommi: ellipses']": 7, "['corr: unknown']": 7, "['orth: 26 missing or incorrect use of apostrophes']": 6, "['orth: Unknown']": 6, "['infl: others']": 5, "['corr: incorrect_adjunctor']": 4, "['orth: 01 lcp instead of cap: polite form']": 4, "['corr: other']": 3, "['woor: deviating']": 3, "['orth: 14 omissions: of one grapheme', 'orth: 17 insertions: other cases']": 2, "['orth: 06 sep instead of tog: compounds', 'orth: 17 insertions: other cases']": 2, "['orth: 02 lcp instead of cap: other cases', 'orth: 10 missing or unnecessary hyphen']": 2, "['corr: incorrect_pronominal_adverb']": 2, "['orth: 02 lcp instead of cap: other cases', 'orth: 07 sep instead of tog: other cases']": 2, "['woor: v1_instead_of_v2']": 1, "['corr: ambiguous_reference']": 1, "['orth: 02 lcp instead of cap: other cases', 'orth: 10 missing or unnecessary hyphen', 'orth: 03 cap instead of lcp']": 1, "['corr: incorrect_preposition_(other_complements)']": 1, "['orth: 17 insertions: other cases', 'orth: 19 transpositions: of (e/ä) / (ä/e)']": 1, "['orth: 07 sep instead of tog: other cases', 'orth: 02 lcp instead of cap: other cases']": 1, "['orth: 14 omissions: of one grapheme', 'orth: 23 transpositions: of consonants: other cases']": 1, "['orth: 14 omissions: of one grapheme', 'orth: 02 lcp instead of cap: other cases']": 1, "['orth: 15 insertions: of double consonants', 'orth: 06 sep instead of tog: compounds']": 1, "['orth: 11 omissions: of double consonants', 'orth: 14 omissions: of one grapheme', 'orth: 13 omissions: of graphemecluster']": 1, "['orth: 06 sep instead of tog: compounds', 'orth: 16 insertions: of a vowel: (ie) instead of (i)']": 1, "['orth: 17 insertions: other cases', 'orth: 15 insertions: of double consonants']": 1, "['infl: weak_instead_of_strong']": 1, "['orth: 23 transpositions: of consonants: other cases', 'orth: 14 omissions: of one grapheme']": 1, "['orth: 02 lcp instead of cap: other cases', 'orth: 14 omissions: of one grapheme']": 1, "['orth: 10 missing or unnecessary hyphen', 'orth: 06 sep instead of tog: compounds', 'orth: 16 insertions: of a vowel: (ie) instead of (i)']": 1, "['orth: 06 sep instead of tog: compounds', 'orth: 18 transpositions: of vowels']": 1, "['woor: other']": 1, "['orth: 19 transpositions: of (e/ä) / (ä/e)', 'orth: 12 omissions: of a vowel: (i) instead of (ie)']": 1, "['orth: 10 missing or unnecessary hyphen', 'orth: 02 lcp instead of cap: other cases']": 1, "['woor: v2_instead_of_vl']": 1, "['orth: 12 omissions: of a vowel: (i) instead of (ie)', 'orth: 15 insertions: of double consonants']": 1, "['orth: 11 omissions: of double consonants', 'orth: 15 insertions: of double consonants']": 1, "['orth: 14 omissions: of one grapheme', 'orth: 06 sep instead of tog: compounds', 'orth: 05 lcp/cap in combination with sep instead of tog/tog instead of sep']": 1, "['orth: 12 omissions: of a vowel: (i) instead of (ie)', 'orth: 17 insertions: other cases']": 1, "['infl: comparision']": 1}
# {'ID2135.txt': 46, 'ID1387.txt': 36, 'ID1113.txt': 36, 'ID1453.txt': 36, 'ID1530.txt': 30, 'ID2960.txt': 30, 'ID1454.txt': 28, 'ID2639.txt': 26, 'ID2892.txt': 25, 'ID2502.txt': 23, 'ID2478.txt': 23, 'ID1316.txt': 22, 'ID2127.txt': 21, 'ID2599.txt': 19, 'ID2785.txt': 18, 'ID1015.txt': 18, 'ID1639.txt': 17, 'ID2105.txt': 17, 'ID2985.txt': 17, 'ID2672.txt': 17, 'ID1626.txt': 16, 'ID1690.txt': 16, 'ID1318.txt': 16, 'ID1470.txt': 15, 'ID2787.txt': 15, 'ID2373.txt': 15, 'ID1246.txt': 14, 'ID1880.txt': 14, 'ID1079.txt': 14, 'ID1296.txt': 14, 'ID2258.txt': 14, 'ID2911.txt': 14, 'ID1157.txt': 14, 'ID2534.txt': 13, 'ID1897.txt': 13, 'ID1713.txt': 13, 'ID2356.txt': 13, 'ID2928.txt': 12, 'ID2820.txt': 12, 'ID1744.txt': 12, 'ID2658.txt': 12, 'ID1490.txt': 12, 'ID2601.txt': 11, 'ID2306.txt': 11, 'ID2319.txt': 11, 'ID1017.txt': 11, 'ID1760.txt': 11, 'ID1375.txt': 10, 'ID1477.txt': 10, 'ID2677.txt': 10, 'ID2065.txt': 10, 'ID2237.txt': 10, 'ID1581.txt': 10, 'ID1399.txt': 10, 'ID1099.txt': 9, 'ID2595.txt': 9, 'ID2223.txt': 9, 'ID2431.txt': 9, 'ID2750.txt': 9, 'ID2255.txt': 9, 'ID2970.txt': 9, 'ID2435.txt': 9, 'ID1646.txt': 9, 'ID2055.txt': 8, 'ID1913.txt': 8, 'ID2506.txt': 8, 'ID1400.txt': 8, 'ID1254.txt': 8, 'ID1156.txt': 8, 'ID2533.txt': 8, 'ID1983.txt': 8, 'ID2235.txt': 8, 'ID2043.txt': 8, 'ID1076.txt': 7, 'ID2761.txt': 7, 'ID2031.txt': 7, 'ID1801.txt': 7, 'ID2398.txt': 7, 'ID1607.txt': 7, 'ID1457.txt': 7, 'ID1597.txt': 7, 'ID2936.txt': 7, 'ID2388.txt': 7, 'ID1140.txt': 7, 'ID2554.txt': 7, 'ID1781.txt': 6, 'ID1384.txt': 6, 'ID1062.txt': 6, 'ID1237.txt': 6, 'ID2777.txt': 6, 'ID1022.txt': 6, 'ID1837.txt': 6, 'ID1675.txt': 6, 'ID2238.txt': 6, 'ID1461.txt': 6, 'ID1364.txt': 5, 'ID1947.txt': 5, 'ID1934.txt': 5, 'ID1289.txt': 5, 'ID2692.txt': 5, 'ID2730.txt': 5, 'ID1253.txt': 5, 'ID1370.txt': 5, 'ID2275.txt': 5, 'ID2305.txt': 5, 'ID1830.txt': 5, 'ID1358.txt': 5, 'ID2345.txt': 5, 'ID1862.txt': 5, 'ID1478.txt': 5, 'ID1701.txt': 4, 'ID1789.txt': 4, 'ID2113.txt': 4, 'ID1653.txt': 4, 'ID2441.txt': 4, 'ID2267.txt': 4, 'ID2847.txt': 4, 'ID1325.txt': 4, 'ID2253.txt': 4, 'ID2608.txt': 4, 'ID1613.txt': 4, 'ID1058.txt': 4, 'ID1373.txt': 3, 'ID2899.txt': 3, 'ID2181.txt': 3, 'ID2841.txt': 3, 'ID2487.txt': 3, 'ID2910.txt': 3, 'ID1366.txt': 3, 'ID2120.txt': 3, 'ID1360.txt': 3, 'ID1925.txt': 3, 'ID1090.txt': 2, 'ID2706.txt': 2, 'ID1137.txt': 2, 'ID2705.txt': 2, 'ID2465.txt': 2, 'ID1255.txt': 2, 'ID1879.txt': 2, 'ID1885.txt': 2, 'ID2202.txt': 2, 'ID2277.txt': 2, 'ID1361.txt': 1, 'ID2509.txt': 1, 'ID1809.txt': 1, 'ID1345.txt': 1, 'ID1604.txt': 1, 'ID2767.txt': 1}
# FOLD: fold6
# {"['orth: 02 lcp instead of cap: other cases']": 267, "['orth: 03 cap instead of lcp']": 156, "['orth: 06 sep instead of tog: compounds']": 151, "['orth: 11 omissions: of double consonants']": 138, "['corr: case']": 84, "['orth: 14 omissions: of one grapheme']": 75, "['orth: 07 sep instead of tog: other cases']": 59, "['orth: 15 insertions: of double consonants']": 55, "['orth: 17 insertions: other cases']": 45, "['orth: 08 tog instead of sep: minimal phraseologism']": 44, "['orth: 23 transpositions: of consonants: other cases']": 44, "['orth: 09 tog instead of sep: other cases']": 38, "['orth: 28 Eigennamen']": 35, "['corr: number']": 34, "['orth: 22 transpositions: of consonants: Fortis/Lenis']": 30, "['orth: 04 lcp/cap behind punctuation marks']": 25, "['corr: gender']": 24, "['ommi: incomplete_clause']": 20, "['orth: 18 transpositions: of vowels']": 16, "['orth: 12 omissions: of a vowel: (i) instead of (ie)']": 15, "['orth: 19 transpositions: of (e/ä) / (ä/e)']": 15, "['orth: 21 transpositions: of vowel and umlaut']": 14, "['corr: inflection_paradigm']": 13, "['corr: incorrect_preposition_(po)']": 13, "['orth: 06 sep instead of tog: compounds', 'orth: 05 lcp/cap in combination with sep instead of tog/tog instead of sep']": 12, "['orth: 27 abbreviations']": 12, "['ommi: ellipses']": 12, "['orth: 10 missing or unnecessary hyphen']": 12, "['infl: others']": 11, "['infl: strong_instead_of_weak']": 10, "['orth: 16 insertions: of a vowel: (ie) instead of (i)']": 9, "['ommi: incomplete_phrase']": 9, "['infl: comparision']": 8, "['corr: unknown']": 8, "['orth: 13 omissions: of graphemecluster']": 7, "['orth: 24 transpositions: of (ss) and (ß)']": 7, "['orth: 26 missing or incorrect use of apostrophes']": 7, "['orth: Unknown']": 7, "['orth: 05 lcp/cap in combination with sep instead of tog/tog instead of sep', 'orth: 06 sep instead of tog: compounds']": 6, "['orth: 02 lcp instead of cap: other cases', 'orth: 10 missing or unnecessary hyphen']": 4, "['woor: deviating']": 3, "['corr: incorrect_preposition_(other_complements)']": 2, "['orth: 22 transpositions: of consonants: Fortis/Lenis', 'orth: 14 omissions: of one grapheme']": 2, "['corr: other']": 2, "['orth: 22 transpositions: of consonants: Fortis/Lenis', 'orth: 11 omissions: of double consonants']": 2, "['orth: 15 insertions: of double consonants', 'orth: 23 transpositions: of consonants: other cases']": 2, "['orth:']": 2, "['corr: incorrect_adjunctor']": 2, "['woor: v1_instead_of_v2']": 2, "['orth: 19 transpositions: of (e/ä) / (ä/e)', 'orth: 23 transpositions: of consonants: other cases']": 2, "['woor: other']": 2, "['orth: 09 tog instead of sep: other cases', 'orth: 03 cap instead of lcp']": 1, "['orth: 07 sep instead of tog: other cases', 'orth: 06 sep instead of tog: compounds']": 1, "['orth: 15 insertions: of double consonants', 'orth: 03 cap instead of lcp']": 1, "['orth: 03 cap instead of lcp', 'orth: 14 omissions: of one grapheme']": 1, "['orth: 16 insertions: of a vowel: (ie) instead of (i)', 'orth: 03 cap instead of lcp']": 1, "['orth: 20 transpositions: of (i/e) / (e/i)']": 1, "['orth: 15 insertions: of double consonants', 'orth: 11 omissions: of double consonants']": 1, "['orth: 11 omissions: of double consonants', 'orth: 02 lcp instead of cap: other cases']": 1, "['orth: 19 transpositions: of (e/ä) / (ä/e)', 'orth: 11 omissions: of double consonants']": 1, "['orth: 09 tog instead of sep: other cases', 'orth: 14 omissions: of one grapheme']": 1, "['orth: 10 missing or unnecessary hyphen', 'orth: 02 lcp instead of cap: other cases', 'orth: Unknown']": 1, "['orth: 03 cap instead of lcp', 'orth: 10 missing or unnecessary hyphen']": 1, "['orth: 22 transpositions: of consonants: Fortis/Lenis', 'orth: 19 transpositions: of (e/ä) / (ä/e)']": 1, "['woor: v2_instead_of_vl']": 1, "['orth: 02 lcp instead of cap: other cases', 'orth: 14 omissions: of one grapheme']": 1, "['orth: 03 cap instead of lcp', 'orth: 17 insertions: other cases', 'orth: 14 omissions: of one grapheme']": 1, "['orth: 06 sep instead of tog: compounds', 'orth: 14 omissions: of one grapheme']": 1, "['orth: 05 lcp/cap in combination with sep instead of tog/tog instead of sep', 'orth: 23 transpositions: of consonants: other cases', 'orth: 06 sep instead of tog: compounds']": 1, "['orth: 23 transpositions: of consonants: other cases', 'orth: 11 omissions: of double consonants']": 1, "['orth: 15 insertions: of double consonants', 'orth: 04 lcp/cap behind punctuation marks']": 1, "['orth: 06 sep instead of tog: compounds', 'orth: 11 omissions: of double consonants']": 1, "['orth: 02 lcp instead of cap: other cases', 'orth: 22 transpositions: of consonants: Fortis/Lenis']": 1, "['orth: 14 omissions: of one grapheme', 'orth: 02 lcp instead of cap: other cases']": 1, "['orth: 02 lcp instead of cap: other cases', 'orth: 23 transpositions: of consonants: other cases']": 1, "['orth: 11 omissions: of double consonants', 'orth: 23 transpositions: of consonants: other cases', 'orth: 12 omissions: of a vowel: (i) instead of (ie)']": 1, "['orth: 12 omissions: of a vowel: (i) instead of (ie)', 'orth: 11 omissions: of double consonants']": 1, "['orth: 02 lcp instead of cap: other cases', 'orth: 13 omissions: of graphemecluster']": 1, "['corr: incorrect_pronominal_adverb']": 1, "['orth: 18 transpositions: of vowels', 'orth: 23 transpositions: of consonants: other cases']": 1, "['orth: 10 missing or unnecessary hyphen', 'orth: 02 lcp instead of cap: other cases', 'orth: 23 transpositions: of consonants: other cases']": 1, "['orth: 15 insertions: of double consonants', 'orth: 14 omissions: of one grapheme', 'orth: 19 transpositions: of (e/ä) / (ä/e)']": 1, "['orth: 17 insertions: other cases', 'orth: 06 sep instead of tog: compounds']": 1, "['infl: weak_instead_of_strong']": 1, "['orth: 14 omissions: of one grapheme', 'orth: 22 transpositions: of consonants: Fortis/Lenis']": 1}
# {'ID2366.txt': 80, 'ID2298.txt': 73, 'ID2015.txt': 34, 'ID1731.txt': 34, 'ID2018.txt': 32, 'ID1906.txt': 31, 'ID2021.txt': 31, 'ID2891.txt': 30, 'ID1107.txt': 29, 'ID2602.txt': 28, 'ID1094.txt': 26, 'ID1722.txt': 26, 'ID1538.txt': 24, 'ID2514.txt': 22, 'ID2745.txt': 22, 'ID2359.txt': 21, 'ID1087.txt': 21, 'ID2516.txt': 21, 'ID1585.txt': 20, 'ID2977.txt': 19, 'ID1515.txt': 19, 'ID2528.txt': 19, 'ID1025.txt': 18, 'ID2901.txt': 17, 'ID1995.txt': 17, 'ID2888.txt': 16, 'ID1363.txt': 16, 'ID2284.txt': 16, 'ID2472.txt': 16, 'ID2154.txt': 15, 'ID1164.txt': 15, 'ID2019.txt': 15, 'ID2561.txt': 15, 'ID1724.txt': 15, 'ID2075.txt': 14, 'ID2875.txt': 14, 'ID2368.txt': 14, 'ID2278.txt': 13, 'ID2547.txt': 13, 'ID1886.txt': 13, 'ID2822.txt': 13, 'ID2086.txt': 13, 'ID2845.txt': 13, 'ID2584.txt': 12, 'ID2066.txt': 12, 'ID2281.txt': 12, 'ID1727.txt': 12, 'ID2842.txt': 12, 'ID1966.txt': 12, 'ID1487.txt': 11, 'ID2585.txt': 11, 'ID1339.txt': 11, 'ID1543.txt': 11, 'ID1942.txt': 11, 'ID2163.txt': 10, 'ID2424.txt': 10, 'ID2952.txt': 10, 'ID2101.txt': 10, 'ID1449.txt': 10, 'ID2429.txt': 10, 'ID1917.txt': 9, 'ID1225.txt': 9, 'ID1041.txt': 9, 'ID2462.txt': 9, 'ID2838.txt': 9, 'ID1919.txt': 9, 'ID1038.txt': 9, 'ID2134.txt': 9, 'ID2923.txt': 8, 'ID2887.txt': 8, 'ID2474.txt': 8, 'ID1278.txt': 8, 'ID2776.txt': 8, 'ID2793.txt': 8, 'ID1273.txt': 8, 'ID2752.txt': 8, 'ID1362.txt': 8, 'ID1589.txt': 8, 'ID1495.txt': 7, 'ID1416.txt': 7, 'ID2330.txt': 7, 'ID1843.txt': 7, 'ID1450.txt': 7, 'ID2683.txt': 7, 'ID2825.txt': 7, 'ID1101.txt': 6, 'ID2333.txt': 6, 'ID1823.txt': 6, 'ID2196.txt': 6, 'ID1057.txt': 6, 'ID1077.txt': 6, 'ID2396.txt': 6, 'ID1623.txt': 6, 'ID1905.txt': 6, 'ID2137.txt': 6, 'ID2890.txt': 6, 'ID2756.txt': 6, 'ID2302.txt': 6, 'ID2962.txt': 5, 'ID2697.txt': 5, 'ID2689.txt': 5, 'ID1876.txt': 5, 'ID2524.txt': 5, 'ID2773.txt': 5, 'ID2902.txt': 5, 'ID2297.txt': 4, 'ID2457.txt': 4, 'ID1524.txt': 4, 'ID2663.txt': 4, 'ID1465.txt': 4, 'ID2712.txt': 4, 'ID1608.txt': 4, 'ID1439.txt': 4, 'ID2394.txt': 4, 'ID1804.txt': 4, 'ID1844.txt': 4, 'ID1222.txt': 4, 'ID1467.txt': 4, 'ID2259.txt': 4, 'ID1959.txt': 4, 'ID1207.txt': 4, 'ID2850.txt': 4, 'ID1365.txt': 4, 'ID1849.txt': 4, 'ID1034.txt': 3, 'ID2670.txt': 3, 'ID1108.txt': 3, 'ID1349.txt': 3, 'ID1615.txt': 3, 'ID1036.txt': 3, 'ID2161.txt': 3, 'ID1650.txt': 3, 'ID1288.txt': 3, 'ID1622.txt': 3, 'ID2307.txt': 3, 'ID1647.txt': 3, 'ID2324.txt': 2, 'ID2206.txt': 2, 'ID1883.txt': 2, 'ID2473.txt': 2, 'ID1368.txt': 2, 'ID2028.txt': 2, 'ID2246.txt': 2, 'ID2254.txt': 2, 'ID1513.txt': 2, 'ID2736.txt': 1, 'ID1818.txt': 1, 'ID1792.txt': 1, 'ID1314.txt': 1}
# FOLD: fold7
# {"['orth: 02 lcp instead of cap: other cases']": 254, "['orth: 11 omissions: of double consonants']": 174, "['orth: 03 cap instead of lcp']": 162, "['orth: 06 sep instead of tog: compounds']": 107, "['orth: 14 omissions: of one grapheme']": 94, "['orth: 07 sep instead of tog: other cases']": 67, "['corr: case']": 61, "['orth: 28 Eigennamen']": 58, "['orth: 22 transpositions: of consonants: Fortis/Lenis']": 46, "['orth: 23 transpositions: of consonants: other cases']": 45, "['orth: 15 insertions: of double consonants']": 41, "['orth: 09 tog instead of sep: other cases']": 38, "['orth: 08 tog instead of sep: minimal phraseologism']": 36, "['orth: 21 transpositions: of vowel and umlaut']": 36, "['orth: 17 insertions: other cases']": 30, "['orth: 04 lcp/cap behind punctuation marks']": 27, "['corr: number']": 27, "['orth: 19 transpositions: of (e/ä) / (ä/e)']": 22, "['corr: gender']": 21, "['corr: inflection_paradigm']": 19, "['infl: others']": 17, "['orth: 27 abbreviations']": 16, "['orth: 16 insertions: of a vowel: (ie) instead of (i)']": 16, "['orth: 12 omissions: of a vowel: (i) instead of (ie)']": 16, "['ommi: incomplete_clause']": 13, "['ommi: incomplete_phrase']": 12, "['orth: 18 transpositions: of vowels']": 12, "['corr: unknown']": 12, "['orth: 26 missing or incorrect use of apostrophes']": 10, "['orth: 01 lcp instead of cap: polite form']": 10, "['corr: incorrect_preposition_(po)']": 10, "['orth: 24 transpositions: of (ss) and (ß)']": 8, "['infl: strong_instead_of_weak']": 8, "['orth: 13 omissions: of graphemecluster']": 7, "['orth: 10 missing or unnecessary hyphen']": 7, "['ommi: ellipses']": 6, "['orth: Unknown']": 5, "['corr: other']": 4, "['orth: 06 sep instead of tog: compounds', 'orth: 05 lcp/cap in combination with sep instead of tog/tog instead of sep']": 4, "['orth: 05 lcp/cap in combination with sep instead of tog/tog instead of sep', 'orth: 06 sep instead of tog: compounds']": 4, "['orth: 02 lcp instead of cap: other cases', 'orth: 10 missing or unnecessary hyphen']": 4, "['orth: 10 missing or unnecessary hyphen', 'orth: 02 lcp instead of cap: other cases']": 4, "['orth: 14 omissions: of one grapheme', 'orth: 17 insertions: other cases']": 3, "['woor: v2_instead_of_vl']": 3, "['corr: incorrect_pronominal_adverb']": 3, "['orth: 21 transpositions: of vowel and umlaut', 'orth: 14 omissions: of one grapheme']": 2, "['orth: 22 transpositions: of consonants: Fortis/Lenis', 'orth: 17 insertions: other cases']": 2, "['orth: 17 insertions: other cases', 'orth: 06 sep instead of tog: compounds']": 2, "['woor: deviating']": 2, "['woor: other']": 2, "['orth: 15 insertions: of double consonants', 'orth: 03 cap instead of lcp']": 2, "['corr: incorrect_adjunctor']": 2, "['woor: vl_instead_of_v2']": 1, "['orth: 22 transpositions: of consonants: Fortis/Lenis', 'orth: 21 transpositions: of vowel and umlaut']": 1, "['orth: 17 insertions: other cases', 'orth: 23 transpositions: of consonants: other cases']": 1, "['orth: 23 transpositions: of consonants: other cases', 'orth: 14 omissions: of one grapheme']": 1, "['orth: 14 omissions: of one grapheme', 'orth: 17 insertions: other cases', 'orth: 02 lcp instead of cap: other cases']": 1, "['orth: 09 tog instead of sep: other cases', 'orth: 13 omissions: of graphemecluster']": 1, "['orth: 22 transpositions: of consonants: Fortis/Lenis', 'orth: 02 lcp instead of cap: other cases']": 1, "['orth: 14 omissions: of one grapheme', 'orth: 02 lcp instead of cap: other cases']": 1, "['orth: 09 tog instead of sep: other cases', 'orth: 15 insertions: of double consonants']": 1, "['orth: 18 transpositions: of vowels', 'orth: 03 cap instead of lcp', 'orth: 11 omissions: of double consonants']": 1, "['orth: 14 omissions: of one grapheme', 'orth: 22 transpositions: of consonants: Fortis/Lenis', 'orth: 18 transpositions: of vowels', 'orth: 03 cap instead of lcp']": 1, "['orth: 14 omissions: of one grapheme', 'orth: 12 omissions: of a vowel: (i) instead of (ie)']": 1, "['orth: 13 omissions: of graphemecluster', 'orth: 03 cap instead of lcp']": 1, "['orth: 11 omissions: of double consonants', 'orth: 14 omissions: of one grapheme']": 1, "['orth: 10 missing or unnecessary hyphen', 'orth: 01 lcp instead of cap: polite form']": 1, "['corr: incorrect_preposition_(other_complements)']": 1, "['orth: 11 omissions: of double consonants', 'orth: 02 lcp instead of cap: other cases']": 1, "['orth: 19 transpositions: of (e/ä) / (ä/e)', 'orth: 02 lcp instead of cap: other cases']": 1, "['orth: 03 cap instead of lcp', 'orth: 10 missing or unnecessary hyphen']": 1, "['orth: 19 transpositions: of (e/ä) / (ä/e)', 'orth: 14 omissions: of one grapheme']": 1, "['orth: 25 incorrect positioning of two graphemes']": 1, "['orth: 07 sep instead of tog: other cases', 'orth: 17 insertions: other cases']": 1, "['orth: 05 lcp/cap in combination with sep instead of tog/tog instead of sep']": 1, "['infl: comparision']": 1, "['woor: v1_instead_of_v2']": 1, "['corr: ambiguous_pronominal_adverb']": 1, "['orth: 14 omissions: of one grapheme', 'orth: 03 cap instead of lcp', 'orth: 02 lcp instead of cap: other cases']": 1, "['orth: 03 cap instead of lcp', 'orth: 14 omissions: of one grapheme']": 1, "['orth: 03 cap instead of lcp', 'orth: 18 transpositions: of vowels']": 1, "['orth: 18 transpositions: of vowels', 'orth: 03 cap instead of lcp']": 1, "['orth: 14 omissions: of one grapheme', 'orth: 13 omissions: of graphemecluster']": 1, "['orth: 13 omissions: of graphemecluster', 'orth: 02 lcp instead of cap: other cases']": 1, "['orth: 15 insertions: of double consonants', 'orth: 06 sep instead of tog: compounds']": 1, "['orth: 17 insertions: other cases', 'orth: 22 transpositions: of consonants: Fortis/Lenis']": 1, "['orth: 10 missing or unnecessary hyphen', 'orth: 07 sep instead of tog: other cases']": 1, "['infl: weak_instead_of_strong']": 1, "['orth: 06 sep instead of tog: compounds', 'orth: 04 lcp/cap behind punctuation marks']": 1, "['orth: 10 missing or unnecessary hyphen', 'orth: 03 cap instead of lcp']": 1, "['orth: 11 omissions: of double consonants', 'orth: 03 cap instead of lcp']": 1, "['orth: 21 transpositions: of vowel and umlaut', 'orth: 10 missing or unnecessary hyphen']": 1, "['orth: 08 tog instead of sep: minimal phraseologism', 'orth: 23 transpositions: of consonants: other cases']": 1, "['orth: 11 omissions: of double consonants', 'orth: 12 omissions: of a vowel: (i) instead of (ie)']": 1, "['orth: 23 transpositions: of consonants: other cases', 'orth: 18 transpositions: of vowels']": 1, "['orth: 06 sep instead of tog: compounds', 'orth: 21 transpositions: of vowel and umlaut']": 1, "['orth: 02 lcp instead of cap: other cases', 'orth: 11 omissions: of double consonants']": 1}
# {'ID2291.txt': 116, 'ID2367.txt': 80, 'ID2211.txt': 65, 'ID2821.txt': 53, 'ID1733.txt': 48, 'ID2114.txt': 36, 'ID2421.txt': 33, 'ID2861.txt': 29, 'ID2543.txt': 27, 'ID2946.txt': 27, 'ID1402.txt': 27, 'ID2355.txt': 26, 'ID2016.txt': 25, 'ID1227.txt': 24, 'ID1377.txt': 23, 'ID1840.txt': 23, 'ID2365.txt': 23, 'ID1864.txt': 22, 'ID1306.txt': 22, 'ID1610.txt': 19, 'ID1234.txt': 19, 'ID1841.txt': 19, 'ID1816.txt': 17, 'ID2371.txt': 17, 'ID2148.txt': 16, 'ID2131.txt': 16, 'ID2422.txt': 16, 'ID2933.txt': 16, 'ID1322.txt': 15, 'ID2640.txt': 15, 'ID1063.txt': 15, 'ID1662.txt': 14, 'ID1945.txt': 14, 'ID1682.txt': 14, 'ID1972.txt': 13, 'ID1008.txt': 13, 'ID1141.txt': 12, 'ID1261.txt': 12, 'ID1001.txt': 12, 'ID2695.txt': 12, 'ID1749.txt': 11, 'ID2851.txt': 11, 'ID1257.txt': 11, 'ID2098.txt': 11, 'ID2442.txt': 11, 'ID2056.txt': 11, 'ID1521.txt': 11, 'ID2407.txt': 10, 'ID2296.txt': 10, 'ID1684.txt': 10, 'ID2666.txt': 10, 'ID2726.txt': 9, 'ID2209.txt': 9, 'ID1944.txt': 9, 'ID1688.txt': 9, 'ID1281.txt': 9, 'ID2149.txt': 9, 'ID2033.txt': 8, 'ID1592.txt': 8, 'ID1808.txt': 8, 'ID2094.txt': 8, 'ID2048.txt': 8, 'ID2176.txt': 8, 'ID2423.txt': 8, 'ID1324.txt': 8, 'ID2119.txt': 8, 'ID1341.txt': 8, 'ID2693.txt': 8, 'ID2659.txt': 8, 'ID2303.txt': 7, 'ID2210.txt': 7, 'ID2203.txt': 7, 'ID1890.txt': 7, 'ID1448.txt': 7, 'ID2070.txt': 7, 'ID1591.txt': 7, 'ID1462.txt': 7, 'ID2834.txt': 6, 'ID2111.txt': 6, 'ID1425.txt': 6, 'ID2383.txt': 6, 'ID2481.txt': 6, 'ID1691.txt': 6, 'ID1201.txt': 6, 'ID1529.txt': 6, 'ID2922.txt': 6, 'ID2611.txt': 6, 'ID1909.txt': 6, 'ID1860.txt': 6, 'ID1996.txt': 6, 'ID1277.txt': 6, 'ID1817.txt': 6, 'ID1803.txt': 6, 'ID2178.txt': 6, 'ID1348.txt': 5, 'ID1213.txt': 5, 'ID2248.txt': 5, 'ID2089.txt': 5, 'ID1182.txt': 5, 'ID2160.txt': 5, 'ID2829.txt': 5, 'ID2546.txt': 5, 'ID2286.txt': 5, 'ID1899.txt': 5, 'ID1915.txt': 5, 'ID1889.txt': 5, 'ID1984.txt': 5, 'ID2189.txt': 5, 'ID1694.txt': 4, 'ID1783.txt': 4, 'ID1098.txt': 4, 'ID2162.txt': 4, 'ID2076.txt': 4, 'ID1193.txt': 4, 'ID1005.txt': 4, 'ID2486.txt': 4, 'ID1327.txt': 3, 'ID2676.txt': 3, 'ID2006.txt': 3, 'ID1673.txt': 3, 'ID1834.txt': 3, 'ID2986.txt': 3, 'ID2940.txt': 3, 'ID1551.txt': 3, 'ID1112.txt': 3, 'ID1893.txt': 3, 'ID2220.txt': 3, 'ID2329.txt': 3, 'ID1787.txt': 3, 'ID1887.txt': 3, 'ID2047.txt': 2, 'ID2022.txt': 2, 'ID1618.txt': 2, 'ID2860.txt': 2, 'ID2939.txt': 2, 'ID1634.txt': 2, 'ID1125.txt': 2, 'ID2915.txt': 2, 'ID2108.txt': 2, 'ID2660.txt': 2, 'ID1033.txt': 1, 'ID2017.txt': 1, 'ID1824.txt': 1, 'ID2622.txt': 1}
# FOLD: fold8
# {"['orth: 02 lcp instead of cap: other cases']": 221, "['orth: 03 cap instead of lcp']": 181, "['orth: 11 omissions: of double consonants']": 131, "['orth: 06 sep instead of tog: compounds']": 98, "['orth: 14 omissions: of one grapheme']": 71, "['orth: 28 Eigennamen']": 59, "['orth: 07 sep instead of tog: other cases']": 52, "['orth: 15 insertions: of double consonants']": 41, "['corr: case']": 40, "['orth: 23 transpositions: of consonants: other cases']": 38, "['orth: 09 tog instead of sep: other cases']": 34, "['orth: 08 tog instead of sep: minimal phraseologism']": 32, "['orth: 17 insertions: other cases']": 31, "['orth: 21 transpositions: of vowel and umlaut']": 27, "['corr: number']": 27, "['orth: 04 lcp/cap behind punctuation marks']": 25, "['orth: 10 missing or unnecessary hyphen']": 17, "['corr: gender']": 15, "['orth: 22 transpositions: of consonants: Fortis/Lenis']": 12, "['orth: 24 transpositions: of (ss) and (ß)']": 12, "['ommi: incomplete_clause']": 10, "['ommi: incomplete_phrase']": 10, "['ommi: ellipses']": 10, "['orth: 19 transpositions: of (e/ä) / (ä/e)']": 10, "['infl: others']": 9, "['orth: 18 transpositions: of vowels']": 9, "['orth: 27 abbreviations']": 9, "['orth: 12 omissions: of a vowel: (i) instead of (ie)']": 9, "['orth: 26 missing or incorrect use of apostrophes']": 8, "['corr: inflection_paradigm']": 8, "['orth: 16 insertions: of a vowel: (ie) instead of (i)']": 7, "['orth: Unknown']": 7, "['corr: incorrect_preposition_(po)']": 6, "['orth: 13 omissions: of graphemecluster']": 5, "['corr: other']": 5, "['infl: strong_instead_of_weak']": 5, "['orth: 05 lcp/cap in combination with sep instead of tog/tog instead of sep', 'orth: 06 sep instead of tog: compounds']": 5, "['orth: 15 insertions: of double consonants', 'orth: 11 omissions: of double consonants']": 4, "['orth: 06 sep instead of tog: compounds', 'orth: 05 lcp/cap in combination with sep instead of tog/tog instead of sep']": 4, "['corr: unknown']": 3, "['infl: comparision']": 3, "['orth: 11 omissions: of double consonants', 'orth: 15 insertions: of double consonants']": 2, "['woor: deviating']": 2, "['orth: 14 omissions: of one grapheme', 'orth: 06 sep instead of tog: compounds']": 2, "['woor: other']": 2, "['corr: incorrect_adjunctor']": 2, "['orth: 09 tog instead of sep: other cases', 'orth: 14 omissions: of one grapheme']": 2, "['orth: 14 omissions: of one grapheme', 'orth: 17 insertions: other cases']": 1, "['orth: 26 missing or incorrect use of apostrophes', 'orth: 05 lcp/cap in combination with sep instead of tog/tog instead of sep', 'orth: 06 sep instead of tog: compounds']": 1, "['orth: 20 transpositions: of (i/e) / (e/i)']": 1, "['orth: 14 omissions: of one grapheme', 'orth: 03 cap instead of lcp']": 1, "['orth: 12 omissions: of a vowel: (i) instead of (ie)', 'orth: 17 insertions: other cases']": 1, "['orth: 02 lcp instead of cap: other cases', 'orth: 23 transpositions: of consonants: other cases']": 1, "['orth: 23 transpositions: of consonants: other cases', 'orth: 11 omissions: of double consonants']": 1, "['orth: 17 insertions: other cases', 'orth: 02 lcp instead of cap: other cases']": 1, "['corr: incorrect_pronominal_adverb']": 1, "['orth: 06 sep instead of tog: compounds', 'orth: 14 omissions: of one grapheme']": 1, "['orth: 02 lcp instead of cap: other cases', 'orth: 15 insertions: of double consonants']": 1, "['orth: 26 missing or incorrect use of apostrophes', 'orth: 28 Eigennamen']": 1, "['orth: 03 cap instead of lcp', 'orth: 07 sep instead of tog: other cases']": 1, "['orth: 25 incorrect positioning of two graphemes', 'orth: 14 omissions: of one grapheme']": 1, "['ommi: others']": 1, "['orth: 02 lcp instead of cap: other cases', 'orth: 14 omissions: of one grapheme']": 1, "['orth: 23 transpositions: of consonants: other cases', 'orth: 06 sep instead of tog: compounds']": 1, "['orth: 14 omissions: of one grapheme', 'orth: 07 sep instead of tog: other cases']": 1, "['orth: 01 lcp instead of cap: polite form']": 1, "['corr: incorrect_preposition_(other_complements)']": 1, "['orth: 02 lcp instead of cap: other cases', 'orth: 17 insertions: other cases']": 1}
# {'ID2146.txt': 72, 'ID1704.txt': 31, 'ID1097.txt': 26, 'ID1185.txt': 26, 'ID2083.txt': 24, 'ID1475.txt': 23, 'ID2512.txt': 23, 'ID2921.txt': 22, 'ID2358.txt': 22, 'ID1904.txt': 21, 'ID1637.txt': 20, 'ID2085.txt': 19, 'ID2587.txt': 19, 'ID2289.txt': 19, 'ID2560.txt': 18, 'ID2954.txt': 17, 'ID1572.txt': 17, 'ID1102.txt': 16, 'ID2541.txt': 16, 'ID1469.txt': 16, 'ID2937.txt': 16, 'ID2650.txt': 15, 'ID2125.txt': 15, 'ID2412.txt': 15, 'ID2184.txt': 14, 'ID2775.txt': 14, 'ID1150.txt': 14, 'ID1850.txt': 14, 'ID1595.txt': 13, 'ID2124.txt': 13, 'ID2577.txt': 13, 'ID1758.txt': 13, 'ID2883.txt': 12, 'ID1747.txt': 12, 'ID2898.txt': 12, 'ID1011.txt': 12, 'ID1282.txt': 12, 'ID1768.txt': 12, 'ID1215.txt': 12, 'ID1151.txt': 12, 'ID1898.txt': 11, 'ID1891.txt': 11, 'ID1665.txt': 11, 'ID1900.txt': 11, 'ID1491.txt': 11, 'ID1133.txt': 11, 'ID2833.txt': 10, 'ID2245.txt': 10, 'ID2810.txt': 10, 'ID1391.txt': 10, 'ID2201.txt': 10, 'ID1354.txt': 10, 'ID1262.txt': 10, 'ID2535.txt': 10, 'ID1851.txt': 10, 'ID1315.txt': 10, 'ID1080.txt': 9, 'ID1527.txt': 9, 'ID1272.txt': 9, 'ID1342.txt': 9, 'ID2673.txt': 9, 'ID1021.txt': 9, 'ID2451.txt': 9, 'ID1200.txt': 9, 'ID1815.txt': 9, 'ID2855.txt': 8, 'ID1074.txt': 8, 'ID1838.txt': 8, 'ID1741.txt': 8, 'ID2734.txt': 8, 'ID1423.txt': 8, 'ID1936.txt': 7, 'ID2918.txt': 7, 'ID1436.txt': 7, 'ID1960.txt': 7, 'ID1558.txt': 7, 'ID1968.txt': 7, 'ID1927.txt': 7, 'ID1024.txt': 7, 'ID1351.txt': 7, 'ID1105.txt': 7, 'ID2434.txt': 7, 'ID2347.txt': 7, 'ID1509.txt': 6, 'ID1832.txt': 6, 'ID2633.txt': 6, 'ID1110.txt': 6, 'ID2938.txt': 6, 'ID1700.txt': 6, 'ID1117.txt': 6, 'ID1135.txt': 6, 'ID1542.txt': 6, 'ID1559.txt': 6, 'ID1544.txt': 6, 'ID1698.txt': 6, 'ID2550.txt': 6, 'ID1103.txt': 5, 'ID2032.txt': 5, 'ID1985.txt': 5, 'ID1389.txt': 5, 'ID1729.txt': 5, 'ID1359.txt': 5, 'ID1274.txt': 5, 'ID2817.txt': 5, 'ID1651.txt': 5, 'ID2354.txt': 5, 'ID2239.txt': 5, 'ID1629.txt': 5, 'ID1186.txt': 5, 'ID2603.txt': 4, 'ID1948.txt': 4, 'ID2766.txt': 4, 'ID2947.txt': 4, 'ID2469.txt': 4, 'ID2797.txt': 4, 'ID2853.txt': 4, 'ID1955.txt': 3, 'ID1594.txt': 3, 'ID1284.txt': 3, 'ID2175.txt': 3, 'ID2167.txt': 3, 'ID2943.txt': 3, 'ID2353.txt': 3, 'ID1946.txt': 3, 'ID2619.txt': 3, 'ID1220.txt': 3, 'ID2099.txt': 3, 'ID2564.txt': 2, 'ID1064.txt': 2, 'ID2268.txt': 2, 'ID1412.txt': 2, 'ID2548.txt': 2, 'ID2582.txt': 2, 'ID1350.txt': 2, 'ID2815.txt': 2, 'ID1813.txt': 2, 'ID1198.txt': 2, 'ID2540.txt': 2, 'ID2816.txt': 1, 'ID1300.txt': 1, 'ID1822.txt': 1, 'ID2001.txt': 1, 'ID2565.txt': 1, 'ID1858.txt': 1}
# FOLD: fold9
# {"['orth: 02 lcp instead of cap: other cases']": 264, "['orth: 03 cap instead of lcp']": 162, "['orth: 11 omissions: of double consonants']": 133, "['orth: 06 sep instead of tog: compounds']": 111, "['orth: 07 sep instead of tog: other cases']": 69, "['corr: case']": 68, "['orth: 14 omissions: of one grapheme']": 67, "['orth: 28 Eigennamen']": 63, "['orth: 23 transpositions: of consonants: other cases']": 51, "['orth: 09 tog instead of sep: other cases']": 42, "['orth: 08 tog instead of sep: minimal phraseologism']": 40, "['orth: 15 insertions: of double consonants']": 37, "['orth: 17 insertions: other cases']": 34, "['corr: number']": 31, "['corr: gender']": 25, "['orth: 21 transpositions: of vowel and umlaut']": 22, "['orth: 22 transpositions: of consonants: Fortis/Lenis']": 22, "['orth: 27 abbreviations']": 19, "['infl: others']": 18, "['orth: 04 lcp/cap behind punctuation marks']": 17, "['corr: inflection_paradigm']": 15, "['orth: 16 insertions: of a vowel: (ie) instead of (i)']": 14, "['orth: 10 missing or unnecessary hyphen']": 12, "['corr: unknown']": 11, "['corr: incorrect_preposition_(po)']": 11, "['ommi: incomplete_clause']": 11, "['orth: 18 transpositions: of vowels']": 10, "['orth: 24 transpositions: of (ss) and (ß)']": 10, "['orth: 26 missing or incorrect use of apostrophes']": 9, "['orth: 01 lcp instead of cap: polite form']": 9, "['orth: 12 omissions: of a vowel: (i) instead of (ie)']": 8, "['orth: 05 lcp/cap in combination with sep instead of tog/tog instead of sep', 'orth: 06 sep instead of tog: compounds']": 8, "['ommi: incomplete_phrase']": 7, "['orth: 19 transpositions: of (e/ä) / (ä/e)']": 7, "['orth: 06 sep instead of tog: compounds', 'orth: 05 lcp/cap in combination with sep instead of tog/tog instead of sep']": 6, "['infl: strong_instead_of_weak']": 6, "['corr: incorrect_adjunctor']": 5, "['infl: comparision']": 5, "['orth: Unknown']": 5, "['ommi: ellipses']": 4, "['orth: 10 missing or unnecessary hyphen', 'orth: 02 lcp instead of cap: other cases']": 4, "['woor: deviating']": 3, "['woor: v2_instead_of_vl']": 3, "['orth: 14 omissions: of one grapheme', 'orth: 02 lcp instead of cap: other cases']": 2, "['orth: 25 incorrect positioning of two graphemes']": 2, "['orth: 13 omissions: of graphemecluster']": 2, "['orth: 14 omissions: of one grapheme', 'orth: 03 cap instead of lcp']": 2, "['corr: other']": 2, "['corr: incorrect_preposition_(other_complements)']": 2, "['orth: 02 lcp instead of cap: other cases', 'orth: 21 transpositions: of vowel and umlaut']": 1, "['orth: 02 lcp instead of cap: other cases', 'orth: 10 missing or unnecessary hyphen']": 1, "['orth: 03 cap instead of lcp', 'orth: 14 omissions: of one grapheme']": 1, "['corr: person']": 1, "['orth: 25 incorrect positioning of two graphemes', 'orth: 19 transpositions: of (e/ä) / (ä/e)']": 1, "['orth: Unknown', 'orth: 22 transpositions: of consonants: Fortis/Lenis']": 1, "['orth: 17 insertions: other cases', 'orth: 22 transpositions: of consonants: Fortis/Lenis']": 1, "['infl: weak_instead_of_strong']": 1, "['orth: 11 omissions: of double consonants', 'orth: 05 lcp/cap in combination with sep instead of tog/tog instead of sep', 'orth: 06 sep instead of tog: compounds']": 1, "['orth: 19 transpositions: of (e/ä) / (ä/e)', 'orth: 02 lcp instead of cap: other cases']": 1, "['orth: 07 sep instead of tog: other cases', 'orth: 06 sep instead of tog: compounds']": 1, "['orth: 07 sep instead of tog: other cases', 'orth: 08 tog instead of sep: minimal phraseologism']": 1, "['orth: 06 sep instead of tog: compounds', 'orth: 02 lcp instead of cap: other cases']": 1, "['orth: 03 cap instead of lcp', 'orth: 19 transpositions: of (e/ä) / (ä/e)']": 1, "['orth: 10 missing or unnecessary hyphen', 'orth: 18 transpositions: of vowels']": 1, "['orth: 14 omissions: of one grapheme', 'orth: 06 sep instead of tog: compounds']": 1, "['orth: 17 insertions: other cases', 'orth: 19 transpositions: of (e/ä) / (ä/e)']": 1, "['orth: 07 sep instead of tog: other cases', 'orth: 23 transpositions: of consonants: other cases']": 1, "['orth: 20 transpositions: of (i/e) / (e/i)']": 1, "['orth: 19 transpositions: of (e/ä) / (ä/e)', 'orth: 14 omissions: of one grapheme']": 1, "['orth: 18 transpositions: of vowels', 'orth: 14 omissions: of one grapheme']": 1, "['orth: 17 insertions: other cases', 'orth: 11 omissions: of double consonants']": 1, "['orth: 14 omissions: of one grapheme', 'orth: 21 transpositions: of vowel and umlaut']": 1, "['orth: 13 omissions: of graphemecluster', 'orth: 03 cap instead of lcp']": 1, "['orth: 18 transpositions: of vowels', 'orth: 02 lcp instead of cap: other cases', 'orth: 14 omissions: of one grapheme']": 1, "['orth: 09 tog instead of sep: other cases', 'orth: 17 insertions: other cases']": 1, "['orth: 06 sep instead of tog: compounds', 'orth: 17 insertions: other cases']": 1, "['orth: 05 lcp/cap in combination with sep instead of tog/tog instead of sep', 'orth: 09 tog instead of sep: other cases']": 1, "['woor: v1_instead_of_v2']": 1, "['orth: 17 insertions: other cases', 'orth: 14 omissions: of one grapheme']": 1, "['orth: 02 lcp instead of cap: other cases', 'orth: 06 sep instead of tog: compounds']": 1, "['orth: 18 transpositions: of vowels', 'orth: 17 insertions: other cases']": 1, "['orth: 07 sep instead of tog: other cases', 'orth: 21 transpositions: of vowel and umlaut']": 1, "['woor: vl_instead_of_v2']": 1, "['ommi: others']": 1, "['orth: 08 tog instead of sep: minimal phraseologism', 'orth: 14 omissions: of one grapheme']": 1, "['orth: 22 transpositions: of consonants: Fortis/Lenis', 'orth: 25 incorrect positioning of two graphemes']": 1, "['orth: 08 tog instead of sep: minimal phraseologism', 'orth: 18 transpositions: of vowels']": 1, "['orth: 02 lcp instead of cap: other cases', 'orth: 17 insertions: other cases']": 1, "['orth: 02 lcp instead of cap: other cases', 'orth: 07 sep instead of tog: other cases']": 1}
# {'ID2285.txt': 66, 'ID2965.txt': 58, 'ID2654.txt': 31, 'ID1614.txt': 30, 'ID2700.txt': 30, 'ID2690.txt': 28, 'ID1039.txt': 25, 'ID2279.txt': 25, 'ID2953.txt': 25, 'ID2227.txt': 24, 'ID1895.txt': 22, 'ID2824.txt': 20, 'ID2299.txt': 19, 'ID1152.txt': 19, 'ID1473.txt': 19, 'ID1413.txt': 18, 'ID2234.txt': 18, 'ID2503.txt': 18, 'ID1184.txt': 18, 'ID1187.txt': 18, 'ID1153.txt': 17, 'ID2011.txt': 17, 'ID2862.txt': 17, 'ID1158.txt': 16, 'ID1916.txt': 16, 'ID2309.txt': 16, 'ID2667.txt': 15, 'ID1805.txt': 15, 'ID2515.txt': 15, 'ID2095.txt': 15, 'ID2963.txt': 15, 'ID1644.txt': 15, 'ID2966.txt': 15, 'ID2357.txt': 15, 'ID1576.txt': 14, 'ID2308.txt': 14, 'ID2310.txt': 13, 'ID1006.txt': 13, 'ID2931.txt': 13, 'ID1982.txt': 13, 'ID2025.txt': 12, 'ID1918.txt': 12, 'ID1531.txt': 12, 'ID2583.txt': 11, 'ID1710.txt': 11, 'ID2786.txt': 11, 'ID2867.txt': 11, 'ID2934.txt': 11, 'ID2701.txt': 11, 'ID1292.txt': 11, 'ID2827.txt': 11, 'ID1424.txt': 11, 'ID2216.txt': 10, 'ID1648.txt': 10, 'ID1154.txt': 10, 'ID2959.txt': 10, 'ID2511.txt': 10, 'ID1603.txt': 10, 'ID1443.txt': 10, 'ID1640.txt': 10, 'ID2013.txt': 10, 'ID2691.txt': 9, 'ID1195.txt': 9, 'ID1174.txt': 9, 'ID2896.txt': 9, 'ID2987.txt': 9, 'ID1575.txt': 8, 'ID2204.txt': 8, 'ID1485.txt': 8, 'ID1142.txt': 8, 'ID2941.txt': 8, 'ID2684.txt': 8, 'ID1635.txt': 8, 'ID1566.txt': 8, 'ID2157.txt': 8, 'ID2849.txt': 8, 'ID1338.txt': 8, 'ID1836.txt': 8, 'ID1703.txt': 8, 'ID1009.txt': 8, 'ID1441.txt': 7, 'ID1445.txt': 7, 'ID2252.txt': 7, 'ID2679.txt': 7, 'ID2261.txt': 7, 'ID1746.txt': 7, 'ID2370.txt': 7, 'ID1019.txt': 7, 'ID1059.txt': 7, 'ID2193.txt': 7, 'ID2839.txt': 7, 'ID1845.txt': 7, 'ID2494.txt': 7, 'ID1881.txt': 7, 'ID2788.txt': 7, 'ID2185.txt': 7, 'ID2751.txt': 7, 'ID1770.txt': 6, 'ID1632.txt': 6, 'ID2463.txt': 6, 'ID1134.txt': 6, 'ID1371.txt': 6, 'ID1506.txt': 6, 'ID2158.txt': 6, 'ID2177.txt': 6, 'ID2110.txt': 6, 'ID2316.txt': 6, 'ID1810.txt': 6, 'ID2467.txt': 5, 'ID2109.txt': 5, 'ID1258.txt': 5, 'ID2553.txt': 5, 'ID1088.txt': 5, 'ID2605.txt': 5, 'ID1317.txt': 5, 'ID2433.txt': 5, 'ID1432.txt': 5, 'ID2343.txt': 5, 'ID1797.txt': 5, 'ID2741.txt': 4, 'ID2051.txt': 4, 'ID1205.txt': 4, 'ID2122.txt': 4, 'ID1353.txt': 4, 'ID2427.txt': 4, 'ID1583.txt': 3, 'ID2973.txt': 3, 'ID1259.txt': 3, 'ID1961.txt': 3, 'ID1251.txt': 3, 'ID1663.txt': 3, 'ID2437.txt': 3, 'ID1229.txt': 3, 'ID2460.txt': 3, 'ID2746.txt': 3, 'ID1279.txt': 3, 'ID1502.txt': 2, 'ID1668.txt': 2, 'ID1666.txt': 2, 'ID1503.txt': 2, 'ID2606.txt': 2, 'ID2387.txt': 2, 'ID1343.txt': 2, 'ID1050.txt': 2, 'ID1286.txt': 1, 'ID2567.txt': 1, 'ID1611.txt': 1, 'ID2222.txt': 1, 'ID1002.txt': 1}
#

from nltk.tokenize import word_tokenize
def count_tokens_char():
    p = '/Users/katinska/GramCorr/corpora/LearnerCorpora/Koko/cv'
    folds_files = [f for f in os.listdir(p) if 'source.txt' in f]

    c = 0
    tokens = 0
    for file in folds_files:
        with codecs.open(os.path.join(p, file), 'r', encoding='utf-8') as inpf:
            for line in inpf.readlines():
                tokenized = word_tokenize(line)
                tokens += len(tokenized)
                for token in tokenized:
                    c += len(token)
    print('total tokens: ', tokens)
    print('total char: ', c)
    sys.exit()
    er_tokens = 0
    er_char = 0
    er_coord = '/Users/katinska/GramCorr/corpora/LearnerCorpora/Koko/cv/error_coordicates_new.csv'
    with codecs.open(er_coord) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for j, row in enumerate(csv_reader):
            if j == 0:
                continue
            err = row[6].split(' ')
            er_tokens += len(err)
            for e in err:
                er_char += len(e)
    print('total er tokens: ', er_tokens)
    print('total er char: ', er_char)


def count_sugg():
    p = '/Users/katinska/GramCorr/mtensemble/input/folds_with_spell_042021_final'
    # p = '/Users/katinska/GramCorr/mtensemble/input/new_folds_last_042021'

    folds_files = [f for f in os.listdir(p) if f.endswith('csv')]
    errors = []
    corrected_all = 0
    total = 0
    sugg_len = 0
    all = []
    error_cor = []
    for file in folds_files:
        with codecs.open(os.path.join(p, file)) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            for j, row in enumerate(csv_reader):
                if j == 0:
                    continue
                if ' '.join(row) not in all:
                    all.append(row)
                else:
                    continue
                total += 1

                if row[0] not in errors and len(errors):
                    errors.append(row[0])
                    sugg_len += suggections
                    suggections = 1
                    if row[4] == '1':
                        corrected = True
                        if (row[0], corrected) not in error_cor:
                            error_cor.append((row[0], corrected))
                    else:
                        corrected = False
                elif not len(errors):
                    errors.append(row[0])
                    suggections = 1
                    if row[4] == '1':
                        corrected = True
                        if (row[0], corrected) not in error_cor:
                            error_cor.append((row[0], corrected))
                    else:
                        corrected = False
                else:
                    suggections += 1
                    if row[4] == '1':
                        corrected = True
                        if (row[0], corrected) not in error_cor:
                            error_cor.append((row[0], corrected))
                    else:
                        corrected = False
                corrected_all += int(corrected)
    print('total', total)
    print('errors', len(errors))
    print('er corr', len(error_cor))
    # print('corr', len(corrected))
    print('sug len', sugg_len/len(errors))




if __name__ == '__main__':
    # correlation()
    # collect_not_corrected()
    # get_not_corrected_error_dict()
    # get_errors_ensemble()
    # check_type_statistics()
    # check_broken_fold()
    # check_guessers()
    # prepare_stat()
    # rtf_performance()
    # count_unique_errors()
    # calculate_not_corrected()
    # count_type_perf_rf()

    # prepare_err_info()
    # count_tokens_char()
    count_sugg()