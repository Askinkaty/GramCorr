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
import difflib
from pprint import pprint

error_table_file = '../corpora/LearnerCorpora/Koko/cv/error_coordinates.csv'
translated = '../translated/Koko_xml/'
data = '../translate/Koko/word_test'
proc_data_dir = '../translate/Koko/split_processed'


class Pair:
    def __init__(self, line, pair, model):
        self.lines = []
        self.line = line
        self.corrections = []
        self.errors = []
        self.hypotheses = []
        self.new_translation = ''
        self.indices = []
        self.hypotheses_ind = []
        self.source = pair[0][1]
        self.target = pair[1][1]
        self.translation = pair[2].strip()
        self.skipped = False
        self.errors_num = 0
        self.model = model

    def get_hypotheses(self):
        if len(self.source) == 1 and '$ $ $' not in self.source[0]:
            return None
        else:
            self.collect_indices()
            if len(self.new_translation) == 0:
                return None
            if len(self.indices) == 1:
                if self.indices[0][0] == 0 and self.indices[0][1] == 0:
                    self.hypotheses.append('')
                    self.lines.append(self.line)
                if self.indices[0][0] == 0 and self.indices[0][1] == len(self.new_translation):
                    self.hypotheses.append('')
                    self.lines.append(self.line)
            elif not self.indices:
                return None
            for m, f in enumerate(self.indices):
                if m == 0 and f[0] != 0:
                    h_start = 0
                    h_end = f[0]
                    self.hypotheses_ind.append((h_start, h_end))
                    h_start = f[1]
                else:
                    h_start = f[1]
                if m + 1 < len(self.indices):
                    h_end = self.indices[m + 1][0]
                else:
                    h_end = len(self.new_translation)
                if h_start != h_end:
                    self.hypotheses_ind.append((h_start, h_end))
                else:
                    if m + 1 != len(self.indices):
                        self.hypotheses_ind.append((h_start, h_end))
            for hi in self.hypotheses_ind:
                shi = hi[0]
                ehi = hi[1]
                self.hypotheses.append(self.new_translation[shi:ehi].strip())
                self.lines.append(self.line)
            if self.model == '1_gram':
                if len(self.hypotheses) > len(self.errors):
                    if '' in self.hypotheses:
                        self.hypotheses.remove('')
            try:
                assert len(self.errors) == len(self.corrections) == len(self.hypotheses)
            except:
                self.skipped = True
                return None
        return True

    def collect_indices(self):
        end = 0
        self.new_translation = self.translation.replace('\\\"', '"').replace('\\"', '"').replace('\n', '')
        margin = 0
        for j, string in enumerate(self.source):
            string = string.replace('\n', '')
            if string == '':
                continue
            if '$ $ $' not in string:
                if '\ "' in string:
                    string = string.replace('\ "', '"')
                string = re.escape(string).replace('"', '\\"')
                # print('STRING: ', string)
                found_all = list(re.finditer(string, self.new_translation))
                # print('FOUND: ', found_all)
                if found_all:
                    if len(found_all) == 1:
                        start = found_all[0].start()
                        end = found_all[0].end()
                        self.indices.append((start, end))
                    else:
                        for found in found_all:
                            if found.start() >= (end + margin):
                                start = found.start()
                                end = found.end()
                                self.indices.append((start, end))
                                break
            elif '$ $ $' in string and not string.replace('$ $ $', '').replace('\n', ''):
                self.errors_num += 1
                self.errors.append('')
                self.corrections.append(self.target[j].replace('$ $ $', '').strip())
                if not self.indices:
                    self.indices.append((0, 0))
            else:
                self.errors_num += 1
                error = self.source[j].replace('$ $ $', '').strip()
                correct = self.target[j].replace('$ $ $', '').strip()
                if error not in self.new_translation[end:] and correct not in self.new_translation[end:]:
                    margin = 1
                else:
                    if len(error) > len(correct):
                        margin = len(correct)
                    else:
                        margin = len(error)
                self.errors.append(error)
                self.corrections.append(correct)


def collect_lines(source, target, number):
    s_lines = []
    t_lines = []
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
    return s_lines, t_lines


def check_hypothesis(correct, error, types, hypothesis, new_trnsl):
    corrected = False
    valid = False
    new_suggestion = False
    types_result = dict()
    for t in types:
        if t not in types_result:
            types_result[t] = dict()
            types_result[t]['all'] = 1
            types_result[t]['correct'] = 0
        else:
            types_result[t]['all'] += 1
    if '\' s' in hypothesis:
        hypothesis = hypothesis.replace("\' s", "\'s")
    if correct == hypothesis:
        corrected = True
        for t in types:
            types_result[t]['correct'] += 1
    elif error == hypothesis:
        valid = True
    else:
        new_suggestion = True
    return corrected, valid, new_suggestion, types_result


def update_dict(old_d, new_d):
    for key, value in new_d.items():
        if key in old_d:
            old_d[key]['all'] += new_d[key]['all']
            old_d[key]['correct'] += new_d[key]['correct']
        else:
            old_d[key] = new_d[key]
    return old_d


def clean_error(error):
    error = error.replace(' ,', ',').replace(" '", "'").replace(" .", "."). \
        replace(' "', '').replace('\\"', '').replace('" ', ''). \
        replace('\\ ', '').replace(' \\', '').replace('[', '').replace(']', ''). \
        replace('°', '').replace('  ', ' ').replace(' %', '%').strip()
    error = re.sub(r'\s*/\s*', '/', error)
    error = re.sub(r'\s*-\s*', '-', error)
    error = re.sub(r'\s*!\s*', '!', error)
    return error


def eval():
    models = [dir for dir in os.listdir(translated) if os.path.isdir(translated + dir)]
    err_table = pd.read_csv(error_table_file, delimiter='\t')
    err_table.fillna('', inplace=True)
    error_dict = dict()
    all_errors =0
    e = 0

    for model in models[:1]:
        print(f'Model: {model}')
        model_dir = os.path.join(translated, model)
        folds = [dir for dir in os.listdir(model_dir)]
        total_types_result = dict()
        fold_acc = 0
        fold_valid = 0
        errors_total = 0
        total_skipped = 0
        for fold in folds[:1]:
            all_errors = 0
            corrected = 0
            valid = 0
            new_suggestions = 0
            cur_line = 0

            number = int(re.search(r'.*([0-9]+)', fold).group(1))
            trans_file = os.path.join(model_dir, fold + '/train.en.trans.de')
            n_best_file = os.path.join(model_dir, fold + '/train.en.nbest.de')
            trans = list(codecs.open(trans_file, 'r', encoding='utf-8'))  # correction
            n_best = list(codecs.open(n_best_file, 'r', encoding='utf-8'))  # best suggestions of the model
            source_file = os.path.join(proc_data_dir, fold + '/train.en')
            target_file = os.path.join(proc_data_dir, fold + '/train.de')
            source = codecs.open(source_file, 'r', encoding='utf-8')  # text with errors
            target = codecs.open(target_file, 'r', encoding='utf-8')  # reference

            s_lines, t_lines = collect_lines(source, target, number)
            assert len(s_lines) == len(t_lines) == len(trans)
            pairs = zip(s_lines, t_lines, trans)
            for k, p in enumerate(pairs):
                d = difflib.Differ()

                source = p[0][1]
                source = ''.join([el.replace('$ $ $', '').replace('\n', ' ') for el in source]).replace('  ', ' ')
                target = p[1][1]
                target = ''.join([el.replace('$ $ $', '').replace('\n', ' ') for el in target]).replace('  ', ' ')
                if source == target:
                    continue
                translantion = p[2]

                a = difflib.unified_diff(source.split(), target.split())
                # errors = list(d.compare(source.split(), target.split()))
                print(list(a))
                print('SOURCE', source)
                print('TARGET', target)
                # pprint(errors)

                # print('TRANS', translantion)


                print('_______________')

                if k > 20:
                    sys.exit()


def main():
    # get_bleu_score()
    eval()
    # build_out_table()


if __name__ == '__main__':
    main()
