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


error_table_file = '../corpora/LearnerCorpora/Koko/cv/error_coordinates.csv'
# translated = '/Users/katinska/GramCorr/translated/Koko_xml/'
translated = '/Users/katinska/GramCorr/translated/Koko_new/'

tr = '/Users/katinska/GramCorr/translated/Koko_xml/10_gram'
data = '../translate/Koko/word_test'
# data = '/Users/katinska/GramCorr/translated/Koko_xml'
proc_data_dir = '../translate/Koko/split_processed'
error_dict_file = 'error_dict_check_042021.pkl'
type_dict_file = 'type_dict_032021.pkl'
random.seed(42)


def get_bleu_score():
    """
    :return: Bleu (optionally Gleu) score for the translation.
    """

    folds = os.listdir(tr)
    # folds = [dir for dir in os.listdir(tr) if os.path.isdir(tr + dir)]
    print(folds)
    fbleu = 0
    fgleu = 0
    for fold in folds:
        target_file = os.path.join(data, fold + '/train.de')
        trans_file = os.path.join(tr, fold + '/train.en.trans.de')
        target = codecs.open(target_file, 'r', encoding='utf-8')  # reference
        trans = codecs.open(trans_file, 'r', encoding='utf-8')  # correction
        references = []
        hypotheses = []
        for pair in zip(target, trans):
            ref = [pair[0].split()]
            hp = pair[1].split()
            if '10_gram' in tr:
                h1 = ''.join(hp)
                hp = h1.split('_')
            # print(hp)
            references.append(ref)
            hypotheses.append(hp)
            # print(hypotheses)
            # print(references)
            # sys.exit()
        bleu_score = nltk.translate.bleu_score.corpus_bleu(references, hypotheses)
        gleu_score = gleu.corpus_gleu(references, hypotheses)
        fbleu += bleu_score
        fgleu += gleu_score
        print(f'Bleu score for {fold}: {bleu_score}')
        # print(f'Gleu score for {fold}: {gleu_score}')
    av_bleu = fbleu / len(folds)
    av_gleu = fgleu / len(folds)
    print(f'Average bleu score: {av_bleu}')
    # print(f'Average gleu score: {av_gleu}')


def glue_characters(line):
    new_line = line.replace(' ', '')
    new_line = new_line.replace('_', ' ')
    new_line = re.sub(r'\s+', ' ', new_line).strip()
    return new_line

class Pair:
    """
    Class for extracting hypotheses from the translation.
    """
    def __init__(self, pair, char):
        self.corrections = []
        self.errors = []
        self.hypotheses = []
        self.new_translation = ''
        self.indices = []
        self.hypotheses_ind = []
        self.source = pair[0][1]
        self.target = pair[1][1]
        self.char = char
        # print(self.char)
        if self.char:
            self.translation = glue_characters(pair[2].strip())
        else:
            self.translation = pair[2].strip()
        self.skipped = False
        self.errors_num = 0
        self.broken = False

    @staticmethod
    def __is_error_line(line):
        if '$ $ $' in line or '$$$' in line:
            return True
        else:
            return False

    @staticmethod
    def __clean_error_line(line):
        return line.replace('$ $ $', '').replace('$$$', '').strip()

    def get_hypotheses(self):
        if len(self.source) == 1 and not self.__is_error_line(self.source[0]):
            return None
        else:
            # print('SOURCE ', self.source)
            # # print('ERROR', )
            # print('TARGET ', self.target)
            # print('TRANSL ', self.translation)
            self.collect_indices()
            # print('INDICES ', self.indices)
            # print('NEW TRANS ', self.new_translation)
            if len(self.new_translation) == 0:
                # print('2')
                return None
            if not self.indices:
                # print('SOURCE ', self.source)
                # print('ERROR', )
                # print('TARGET ', self.target)
                # print('TRANSL ', self.translation)
                # print('3')
                # sys.exit()
                return None
            if len(self.indices) == 1:
                if self.indices[0][0] == 0 and self.indices[0][1] == 0:
                    self.hypotheses.append('')
                if self.indices[0][0] == 0 and self.indices[0][1] == len(self.new_translation):
                    self.hypotheses.append('')
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
                self.hypotheses.append(self.new_translation[hi[0]:hi[1]].strip())
            try:
                assert len(self.errors) == len(self.corrections) == len(self.hypotheses)
            except:
                # print(self.hypotheses)
                new_hypotheses = []
                if '' in self.errors or '' in self.corrections:
                    for el in self.hypotheses:
                        if el == '' and (el not in self.errors or el not in self.corrections):
                            continue
                        else:
                            new_hypotheses.append(el)
                # if '' in self.hypotheses:
                #     self.hypotheses.remove('')
                try:
                    assert len(self.errors) == len(self.corrections) == len(self.hypotheses)
                    # print('#' * 100)
                    # print('FOUND HYP: ', self.hypotheses)

                    # return None
                    # print('All good!')
                    return True
                except:
                    # try:
                    #     new_hypotheses = []
                    #     for el in self.hypotheses:
                    #         if ' ' in el:
                    #             new_hypotheses.extend(el.split())
                    #         else:
                    #             new_hypotheses.append(el)
                    #     self.hypotheses = new_hypotheses
                    #     print('NEW HYPOTH', self.hypotheses)
                    #     print(new_hypotheses)
                    #     assert len(self.errors) == len(self.corrections) == len(self.hypotheses)
                    #     print('All good!')
                    #     return True
                    # except:
                    # print('SOURCE ', self.source)
                    # print('TARGET ', self.target)
                    # print('TRANSL ', self.translation)
                    # print('ERRORS', self.errors)
                    # print('CORRECTIONS', self.corrections)
                    # print('HYPOTHESES', self.hypotheses)
                    self.skipped = True
                    # print('4')
                    return None
        return True

    def find_strings(self, a, b):
        indx = []
        start = 0
        end = 0
        while end < len(a):
            # print(a)
            # print(b)
            # print('____')
            start = a.find(b, start)
            if start != -1:
                end = start + len(b)
                if (start, end) not in indx:
                    indx.append((start, end))
                    start = end
            else:
                break
        return indx


    def collect_indices(self):
        end = 0
        self.new_translation = self.translation.replace('\\\"', '"').replace('\\"', '"').replace('\n', '')
        # print('NEW TRANSL ', self.new_translation)
        margin = 0
        not_errors = 0
        for j, s in enumerate(self.source):
            s = s.replace('\n', '').replace('\ "', '"')
            if len(s) > 1:
                s = s.strip()
            if s == '':
                continue
            if not self.__is_error_line(s):
                not_errors += 1
                # print('S: ', '|' + s + '|')
                # print('NEW TR: ', self.new_translation)
                # s = re.escape(s).replace('"', '\\"')
                # found_all = list(re.finditer(s, self.new_translation))

                found_all = self.find_strings(self.new_translation, s)
                # print(found_all)
                # if not len(found_all):
                #     st = self.new_translation.find(s)
                #     ed = len(s) + st
                #     self.indices.append((st, ed))
                #     # print(self.indices)
                #     break
                if found_all:
                    # print(found_all)
                    if len(found_all) == 1:
                        # start = found_all[0].start()
                        # end = found_all[0].end()
                        start = found_all[0][0]
                        end = found_all[0][1]
                        self.indices.append((start, end))
                    else:
                        for k, found in enumerate(found_all):

                            # we need it because we could have the same string repeated many times (e.g. 'und')
                            # so we need to know where to start to search
                            if found[0] >= (end + margin):
                                # start = found.start()
                                # end = found.end()
                                start = found[0]
                                end = found[1]
                                self.indices.append((start, end))
                                break
            elif self.__is_error_line(s) and not self.__clean_error_line(s):
                # print('Err line 1', s)
                self.errors_num += 1
                self.errors.append('')
                self.corrections.append(self.__clean_error_line(self.target[j]))
                if not self.indices:
                    self.indices.append((0, 0))
            else:
                # print('Err line 2', s)

                self.errors_num += 1
                error = self.__clean_error_line(self.source[j])
                # print(error)
                correct = self.__clean_error_line(self.target[j])
                # print(correct)
                # print(self.new_translation[end:])
                if error not in self.new_translation[end:] and correct not in self.new_translation[end:]:
                    margin = 1
                else:
                    if len(error) > len(correct):
                        margin = len(correct)
                    else:
                        margin = len(error)
                self.errors.append(error)
                self.corrections.append(correct)
        if len(self.indices) != not_errors:
            self.broken = True


def update_lines(line):
    new_s_line = []
    s = ''
    for i, sl in enumerate(line):
        if not ('$ $ $' in sl or '$$$' in sl):
            s += sl
            if i + 1 == len(line):
                new_s_line.append(s)
                s = ''
        else:
            if s:
                new_s_line.append(s)
            new_s_line.append(sl)
            s = ''
    return new_s_line

def collect_lines(source, target, number, char):
    s_lines = []
    t_lines = []
    z = list(zip(source, target))
    full_s_line = []
    full_t_line = []
    for i, pair in enumerate(z):
        s_line = pair[0]
        t_line = pair[1]
        if '# # #' in s_line:
            new_full_s_line = update_lines(full_s_line)
            new_full_t_line = update_lines(full_t_line)
            # s_lines.append((number, full_s_line))
            # t_lines.append((number, full_t_line))
            s_lines.append((number, new_full_s_line))
            t_lines.append((number, new_full_t_line))
            full_s_line = []
            full_t_line = []
            continue
        else:
            if char:
                s_line = glue_characters(s_line)
                t_line = glue_characters(t_line)
            if ('$ $ $' in s_line or '$$$' in s_line) and i+2 < len(z) and ('$ $ $' in z[i+2][0] or '$$$' in z[i+2][0]):
                full_s_line.append(s_line)
                full_s_line.append(' ')
            else:
                if s_line != '':
                    full_s_line.append(s_line)

            if ('$ $ $' in t_line or '$$$' in t_line) and i+2 < len(z) and ('$ $ $' in z[i+2][1] or '$$$' in z[i+2][1]):
                full_t_line.append(t_line)
                full_t_line.append(' ')
            else:
                if t_line != '':
                    full_t_line.append(t_line)
    return s_lines, t_lines



def check_hypothesis(correct, error, types, hypothesis):
    corrected = False
    not_corrected = False
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
        not_corrected = True
    else:
        new_suggestion = True
    return corrected, not_corrected, new_suggestion, types_result


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


def make_type_report(total_types_result):
    # print(f'Total erros in the data: {errors_total}')
    out_dict = dict()
    for key, value in total_types_result.items():
        type_acc = round((value['correct'] * 100) / value['all'])
        out_dict[key] = type_acc
    result = [(k, out_dict[k]) for k in sorted(out_dict, key=out_dict.get, reverse=True)]
    for k, v in result:
        all_num = total_types_result[k]['all']
        print(f'{k}: {v}%, # of errors: {all_num}')


def get_data(fold, model, char, proc_data_dir):
    Data = namedtuple('Data', ['translation', 'n_best', 'source', 'target'])
    model_dir = os.path.join(translated, model)
    trans_file = os.path.join(model_dir, fold + '/train.en.trans.de')
    n_best_file = os.path.join(model_dir, fold + '/train.en.nbest.de')
    trans = list(codecs.open(trans_file, 'r', encoding='utf-8'))  # correction
    n_best = list(codecs.open(n_best_file, 'r', encoding='utf-8'))  # best suggestions of the model
    # n_best = []
    if char:
        proc_data_dir += '_char'
    source_file = os.path.join(proc_data_dir, fold + '/train.en')
    target_file = os.path.join(proc_data_dir, fold + '/train.de')
    source = codecs.open(source_file, 'r', encoding='utf-8')  # text with errors
    target = codecs.open(target_file, 'r', encoding='utf-8')  # reference
    d = Data(trans, n_best, source, target)
    return d


def get_other_hypotheses(line, source_line, target_line, n_best, correction, error, types, i, char):
    hypotheses = dict()

    # here add in dict scores for every new hypotheses

    score = 0
    hyp_num = 5
    corr = False
    for b_line in n_best[1:]:
        score_line = b_line.split('|||')
        if int(score_line[0]) + 1 == line:
            # if corrected and not score:
            score = float(score_line[-1].strip())
            pair = Pair((source_line, target_line, score_line[1]), char)
            # print(error)
            # print(source_line)
            # print(target_line)
            # print('_______________')
            result = pair.get_hypotheses()
            if not result or pair.broken or pair.skipped:
                # print('Broken in other hypotheses')
                continue
            else:
                # print('passed')
                # corrected, not_corrected, new_suggestion, types_result
                n_c, n_v, new_suggestion, _ = check_hypothesis(correction, error, types,
                                                               pair.hypotheses[i])
                # if new_suggestion:
                #     print(error)
                #     print(pair.hypotheses[i])
                #     print(n_c)
                #     print(score)
                #     print('))))))))))))))')
                #     print('NEW SUGG', new_suggestion)
                if n_c:
                    corr = True
                if pair.hypotheses[i] not in hypotheses.keys():
                    hypotheses[pair.hypotheses[i]] = float(score_line[-1].strip())
                if len(hypotheses) > hyp_num:
                    break
    return hypotheses, score, corr


def eval():
    models = [dir for dir in os.listdir(translated) if os.path.isdir(translated + dir)]
    err_table = pd.read_csv(error_table_file, delimiter='\t')
    err_table.fillna('', inplace=True)
    error_dict = dict()
    missing = 0
    all_av_not_corrected = 0
    all_av_corrected = 0
    print(models)
    for model in models:
        # print(model)
        # if '10' not in model:
        #     continue
        print(f'Model: {model}')
        if model == '10_gram':
            char = True
        else:
            char = False

        model_dir = os.path.join(translated, model)
        folds = [dir for dir in os.listdir(model_dir)]
        total_types_result = dict()
        fold_acc = 0
        fold_not_corrected = 0
        fold_sugg = 0
        errors_total = 0
        total_skipped = 0
        for fold in folds:
            print(fold)
            all_errors = 0
            corrected = 0
            not_corrected = 0
            new_suggestions = 0
            cur_line = 0
            number = int(re.search(r'.*([0-9]+)', fold).group(1))
            data = get_data(fold, model, char, proc_data_dir)
            s_lines, t_lines = collect_lines(data.source, data.target, number, char)
            assert len(s_lines) == len(t_lines) == len(data.translation)
            pairs = zip(s_lines, t_lines, data.translation)
            for k, p in enumerate(pairs):
                cur_line += 1
                # print(p[0])
                # print(p[1])
                # sys.exit()

                pair = Pair(p, char)
                # if not p[0][1][0].startswith('Eine zweifellos'):
                #     continue
                # print('source', pair.source)
                # print('target', pair.target)

                # Insert here iteration over n_best and take the score of the first suggestion (top)
                # save as top_score
                for b_line in data.n_best:
                    score_line = b_line.split('|||')
                    if int(score_line[0]) == cur_line:
                        score = float(score_line[-1].strip())
                        break

                result = pair.get_hypotheses()
                total_skipped += int(pair.skipped)
                # if 'Leider zu oft kommen Situationen vor' in p[1][1]:
                #     sys.exit()
                if not result or pair.broken:

                    # print('RESULT:', result)
                    # print('Broken: ', pair.broken)
                    # print(pair.errors)
                    for l, h in enumerate(pair.errors):
                        # print(h)
                        error = clean_error(pair.errors[l])
                        # print(error)
                        i = err_table.loc[
                            (err_table['fold#'] == 'fold' + str(number)) & (err_table['line'] == cur_line) &
                            (err_table['error'] == error)].index.values.tolist()
                        if i:
                            ind = i[0]
                            row = err_table.loc[ind, :]
                            # print(row)
                            types = ast.literal_eval(row['types'])
                            err_id = '_'.join([str(j) for j in row[:7]])
                            if err_id not in error_dict:
                                error_dict[err_id] = dict()
                                error_dict[err_id]['type'] = types
                            new_hypotheses, other_score, corr = get_other_hypotheses(cur_line, p[0], p[1], data.n_best,
                                                                                 pair.corrections[l], pair.errors[l], types,
                                                                                 l,
                                                                                 char)
                            if not len(new_hypotheses):
                                continue
                            if model not in error_dict[err_id]:
                                error_dict[err_id][model] = dict()
                            error_dict[err_id][model]['other_suggestions'] = new_hypotheses
                            error_dict[err_id][model]['expected'] = pair.corrections[l]
                            error_dict[err_id][model]['other_suggestions_score'] = other_score
                            # print(error_dict[err_id][model])

                else:
                    all_errors += pair.errors_num
                    for l, h in enumerate(pair.errors):
                        error = clean_error(pair.errors[l])
                        i = err_table.loc[(err_table['fold#'] == 'fold' + str(number)) & (err_table['line'] == cur_line) &
                                          (err_table['error'] == error)].index.values.tolist()
                        if i:
                            ind = i[0]
                            row = err_table.loc[ind, :]
                            # print(row)
                            types = ast.literal_eval(row['types'])
                            err_id = '_'.join([str(j) for j in row[:7]])
                            if err_id not in error_dict:
                                error_dict[err_id] = dict()
                                error_dict[err_id]['type'] = types
                            c, v, new_suggestion, types_result = check_hypothesis(pair.corrections[l],
                                                                                  pair.errors[l],
                                                                                  types,
                                                                                  pair.hypotheses[l])
                            if model not in error_dict[err_id]:
                                error_dict[err_id][model] = dict()
                            error_dict[err_id][model]['corrected'] = int(c)
                            error_dict[err_id][model]['not_corrected'] = int(v)
                            error_dict[err_id][model]['new_suggestion/correction'] = pair.hypotheses[l]
                            error_dict[err_id][model]['new_suggection/correction_score'] = score
                            total_types_result = update_dict(total_types_result, types_result)
                            corrected += int(c)
                            new_suggestions += int(new_suggestion)
                            # print('Corrected:', c)
                            new_hypotheses, other_score, corr = get_other_hypotheses(cur_line, p[0], p[1], data.n_best,
                                                                      pair.corrections[l], pair.errors[l], types, l,
                                                                        char)

                            if corr:
                                corrected += int(corr)
                            if not corrected and not corr:
                                not_corrected += int(v)

                            error_dict[err_id][model]['other_suggestions'] = new_hypotheses
                            error_dict[err_id][model]['expected'] = pair.corrections[l]

                            # if c:
                            # error_dict[err_id][model]['correction'] = pair.corrections[l]
                            error_dict[err_id][model]['other_suggestions_score'] = other_score

                            # print(error_dict[err_id][model])
    #         facc = round((corrected * 100) / all_errors)
    #         fnotcorr = round((not_corrected * 100) / all_errors)
    #         fsugg = round((new_suggestions * 100)/all_errors)
    #         fold_acc += facc  # from all errors, how many were corrected as expected?
    #         fold_not_corrected += fnotcorr
    #         fold_sugg += fsugg
    #         errors_total += all_errors
    #         print(f'Accuracy for {fold}: {facc}%')
    #         print(f'Average not corrected at all {fold}: {fnotcorr}%')
    #         print(f'Average new suggestions for {fold}: {fsugg}%')
    #     av_acc = fold_acc / len(folds)
    #     av_not_corr = fold_not_corrected / len(folds)
    #     av_sugg = fold_sugg / len(folds)
    #     all_av_not_corrected += av_not_corr
    #     all_av_corrected += av_acc
    #     print(f'Skipped: {total_skipped}')
    #     print(f'Average fold accuracy: {av_acc}%')
    #     print(f'Average fold not corrected at all : {av_not_corr}%')
    #     print(f'Average fold new suggestions : {av_sugg}%')
    #     # make_type_report(total_types_result)
    # # print(f'Error dict: {error_dict}')
    # print(f'All errors #: {errors_total}')
    # print(f'Missing errors #: {missing}')
    # print(f'All av. not corrected #: {all_av_not_corrected/4}')
    # print(f'All av. corrected #: {all_av_corrected/4}')
    #
    with open(type_dict_file, 'wb') as td:
        pickle.dump(total_types_result, td)
    with open(error_dict_file, 'wb') as f:
        pickle.dump(error_dict, f)


def collect_error_info(error_id, data, models):
    error_info = []
    model_list = []
    error = error_id.split('_')[-1]
    for k, v in data.items():
        if k != 'type':
            model_dict = v
            correction = None
            expected = model_dict['expected']
            if 'corrected' in model_dict and model_dict['corrected'] == 1:
                correction = model_dict['new_suggestion/correction']
                score = model_dict['new_suggection/correction_score']
                corrected = 1
                model_list.append(k)
                error_info.append((expected, k, correction, score, corrected))
            if model_dict['other_suggestions']:
                for suggestion, s_score in model_dict['other_suggestions'].items():
                    if suggestion == error:
                        continue
                    if correction and suggestion == correction and 'corrected' in model_dict:
                        continue
                    if not len(suggestion):
                        continue
                    new_suggestion = (expected, k, suggestion, s_score, int(suggestion == model_dict['expected']))
                    if new_suggestion not in error_info:
                        error_info.append(new_suggestion)
            if 'not corrected' in model_dict and model_dict['not_corrected'] == 1 and not model_dict['other_suggestions']:
                error_info.append((None, k, None, 0.0, 0))
        if k == 'type':
            t = v[0]
    if len(error_info) < 3:
        for model in models:
            if model not in model_list:
                error_info.append((None, model, None, 0.0, 0))
    return error_info, t


def build_rows(error_info, t, error_id, models):
    rows = []
    error = error_id.split('_')[-1]
    # error_info.append((expected, k, correction, score, corrected))
    # (None, model, None, 0.0, 0)
    hyp = None
    for tuple in error_info:
        if tuple[2] is not None and hyp != tuple[2]:
            hyp = tuple[2]
            cor = tuple[4]
            row = [tuple[0], error_id, t, len(error), hyp, cor]
            other_models = OrderedDict()
            for model in models:
                other_models[model] = dict()
            for tpl in error_info:
                other_models[tpl[1]][tpl[2]] = tpl[3]
            # print(other_models)
            for m in other_models:
                if hyp in other_models[m].keys():
                    other_score = other_models[m][hyp]
                    row.append(1)
                    row.append(other_score)
                elif not other_models[m].keys() or None in other_models[m].keys():
                    row.append(0)
                    row.append(0.0)
                else:
                    row.append(0)
                    row.append(0.0)
                    # row.append(-1)
                    # other_score = max(other_models[m].values())
                    # row.append(other_score)
            rows.append(row)
    return rows


def build_out_table():
    models = [dir for dir in os.listdir(translated) if os.path.isdir(translated + dir)]
    path_to_errors = 'error_dict_check_042021.pkl'

    out_file = codecs.open('out_moses_table_042021.csv', 'w')
    out_writer = csv.writer(out_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    c = 0
    corrected = 0
    not_corrected = 0
    already_written = []
    already_recorded_sugg = []
    with open(path_to_errors, 'rb') as pickle_file:
        error_data = pickle.load(pickle_file)
        # print(error_data['fold3_4614_ID1935.txt_18_61_61_Hobbies'])

        # sys.exit()
        header = ['error_id', 'type', 'error_length', 'suggestion', 'is_correct']
        for model in models:
            header.append(model + '_is_suggested')
            header.append(model + '_score')
        out_writer.writerow(header)
        all_errors = []

        for key, value in error_data.items():
            # if key != 'fold3_4614_ID1935.txt_18_61_61_Hobbies':
            #     continue
            c += 1
            error_info, t = collect_error_info(key, value, models)
            # print(error_info)
            corrected += any([el[-1] for el in error_info])
            # print(corrected)
            if not any([el[-1] for el in error_info]):
                not_corrected += 1
                continue
            for row in build_rows(error_info, t, key, models):
                str_row = '_'.join([str(el) for el in row])
                s = row[1] + '_' + row[4]
                if str_row not in already_written and s not in already_recorded_sugg:
                    already_written.append(str_row)
                    already_recorded_sugg.append(s)
                    all_errors.append(row[1:] + [row[0]]) # exclude expected?
                    # all_errors.append(row[1:]) # exclude expected?

            # if c > 30:
            #     break
                    # sys.exit()
        for a_er in all_errors:
            out_writer.writerow(a_er)
    print(not_corrected)
    return len(all_errors), len(error_data.keys()), corrected


def chunks(lst, n):
    l = np.array_split(lst, n)
    return list(l)

def collect_same_er(l):
    out = []
    for er in l:
        er_id = er[1]
        n = len([e for e in l if e[1] == er_id])
        if n > 1:
            new_er = [er for er in l if er[1] == er_id]
            if new_er not in out:
                out.append(new_er)
        else:
            out.append(er)
    return out


def write_folds(folds, models):
    out_dir = '/Users/katinska/GramCorr/mtensemble/input/new_folds_last_042021'
    c = 0
    for i, e in enumerate(folds):
        # print(e)
        random.shuffle(e)
        with open(os.path.join(out_dir, 'fold'+str(i)+'.csv'), mode='w') as out_file:
            writer = csv.writer(out_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            # header = ['expected', 'error_id', 'type', 'error_length', 'suggestion', 'is_correct']
            header = ['error_id', 'type', 'error_length', 'suggestion', 'is_correct']
            for model in models:
                header.append(model + '_is_suggested')
                header.append(model + '_score')
            header.append('expected')
            writer.writerow(header)
            for error in e:
                if error:
                    if not any(isinstance(el, list) for el in error):
                        writer.writerow(error)
                        c += 1
                    else:
                        for hyp in error:
                            c += 1
                            writer.writerow(hyp)
    print(c)
    print('Finished')

def write_fill_data(full, models):
    out_dir = '/Users/katinska/GramCorr/mtensemble/input/new_folds_last_1'
    full_data_file = os.path.join(out_dir, 'data.csv')
    with open(full_data_file, mode='w') as full_out:
        random.shuffle(full)
        fwriter = csv.writer(full_out, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = ['error_id', 'type', 'error_length', 'suggestion', 'is_correct']
        # header = ['expected', 'error_id', 'type', 'error_length', 'suggestion', 'is_correct']
        for model in models:
            header.append(model + '_is_suggested')
            header.append(model + '_score')
        header.append('expected')
        fwriter.writerow(header)
        for f_er in full:
            if f_er:
                if not any(isinstance(el, list) for el in f_er):
                    fwriter.writerow(f_er)
                else:
                    for f_hyp in f_er:
                        fwriter.writerow(f_hyp)


def split_table():
    models = [dir for dir in os.listdir(translated) if os.path.isdir(translated + dir)]
    folds = dict()
    new_folds = [[] for i in range(10)]
    full = []
    with codecs.open('out_moses_table_042021.csv') as table_file:
        next(table_file, None)
        table_reader = csv.reader(table_file, delimiter='\t')
        for row in table_reader:
            fold = row[0].split('_')[0]
            if fold not in folds:
                folds[fold] = []
            folds[fold].append(row) # NB!
    # sys.exit()
    for k, v in folds.items():
        # print(len(v))

        v = collect_same_er(v)
        # print(v)
        # print(len(v))
        random.shuffle(v)
        c = chunks(v, 10)
        for j, chunk in enumerate(c):
            # print(len(chunk))
            new_folds[j].extend(chunk.tolist())
            full.extend(chunk.tolist())
    # print(len(full))
    # # print(full)
    f = []
    for c in full:
        f.extend(c)
    print(len(f))
    write_folds(new_folds, models)
    # # write_fill_data(full, models)


def main():
    # get_bleu_score()
    # eval()
    # all_errors, got_suggections, corrected = build_out_table()
    # print(f'All errors in the data: {all_errors}')
    # print(f'All errors which left after filtering broken: {got_suggections}')
    # print(f'Corrected by at least one system: {corrected}')
    split_table()


if __name__ == '__main__':
    main()

