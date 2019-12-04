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

error_table_file = '../corpora/LearnerCorpora/Koko/cv/error_coordinates.csv'
translated = '../translated/Koko_xml/'
data = '../translate/Koko/word_test'
proc_data_dir = '../translate/Koko/split_processed'
error_dict_file = 'error_dict.pkl'


def get_bleu_score():
    folds = [dir for dir in os.listdir(translated) if os.path.isdir(translated + dir)]
    fbleu = 0
    fgleu = 0
    for fold in folds:
        target_file = os.path.join(data, fold + '/train.de')
        trans_file = os.path.join(translated, fold + '/train.en.trans.de')
        target = codecs.open(target_file, 'r', encoding='utf-8')  # reference
        trans = codecs.open(trans_file, 'r', encoding='utf-8')  # correction
        references = []
        hypotheses = []
        for pair in zip(target, trans):
            ref = [pair[0].split()]
            hp = pair[1].split()
            references.append(ref)
            hypotheses.append(hp)
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

    for model in models:
        print(f'Model: {model}')
        model_dir = os.path.join(translated, model)
        folds = [dir for dir in os.listdir(model_dir)]
        total_types_result = dict()
        fold_acc = 0
        fold_valid = 0
        errors_total = 0
        total_skipped = 0
        for fold in folds:
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
                cur_line += 1
                pair = Pair(cur_line, p, model)
                result = pair.get_hypotheses()
                total_skipped += int(pair.skipped)
                if not result:
                    continue
                else:
                    all_errors += pair.errors_num
                    for l, h in enumerate(pair.errors):
                        line = pair.lines[l]
                        error = clean_error(pair.errors[l])
                        i = err_table.loc[(err_table['fold#'] == 'fold' + str(number)) & (err_table['line'] == line) & (
                                err_table['error'] == error)].index.values.tolist()
                        # if not i:
                        #     print('did not find error')
                        #     print(fold)
                        #     print(number)
                        #     print(line)
                        #     print(error)
                        #     print('#'*100)
                        if i:
                            ind = i[0]
                            row = err_table.loc[ind, :]
                            types = ast.literal_eval(row['types'])
                            err_id = '_'.join([str(j) for j in row[:7]])
                            if err_id not in error_dict:
                                e += 1
                                error_dict[err_id] = dict()
                                error_dict[err_id]['type'] = types
                            c, v, new_suggestion, types_result = check_hypothesis(pair.corrections[l],
                                                                                  pair.errors[l],
                                                                                  types,
                                                                                  pair.hypotheses[l],
                                                                                  pair.new_translation)
                            if model not in error_dict[err_id]:
                                error_dict[err_id][model] = dict()
                            error_dict[err_id][model]['corrected'] = int(c)
                            error_dict[err_id][model]['valid'] = int(v)
                            total_types_result = update_dict(total_types_result, types_result)
                            corrected += int(c)
                            valid += int(v)
                            print('Corrected!', c)
                            new_suggestions += int(new_suggestion)

                            if c:
                                error_dict[err_id][model]['correction'] = pair.corrections[l]
                                for b_line in n_best:
                                    score_line = b_line.split('|||')
                                    if int(score_line[0]) + 1 == line:
                                        score = float(score_line[-1].strip())
                                        print('SCORE ', score)
                                        error_dict[err_id][model]['score'] = float(score_line[-1].strip())
                                        break
                            else:
                                for b_line in n_best:
                                    score_line = b_line.split('|||')
                                    if int(score_line[0]) + 1 == line:
                                        n_pair = Pair(cur_line, (p[0], p[1], score_line[1]), model)
                                        n_result = n_pair.get_hypotheses()
                                        if not n_result or n_pair.skipped:
                                            continue
                                        else:
                                            n_c, n_v, new_suggestion, _ = check_hypothesis(pair.corrections[l],
                                                                                           pair.errors[l],
                                                                                           types,
                                                                                           n_pair.hypotheses[l],
                                                                                           n_pair.new_translation)

                                            if not n_c and n_v:
                                                continue
                                            else:
                                                error_dict[err_id][model]['suggestion'] = n_pair.hypotheses[l]
                                                error_dict[err_id][model]['score'] = float(score_line[-1].strip())
                                                break
            # print('Corrected: ', corrected)
            # print('All errors in fold: ', all_errors)
            facc = round((corrected * 100) / all_errors)
            fvalid = round((valid * 100) / all_errors)
            fold_acc += facc  # from all errors, how many were corrected as expected?
            fold_valid += fvalid
            errors_total += all_errors
            print(f'Accuracy for {fold}: {facc}%')
            print(f'Average not valid for {fold}: {100-fvalid}%')
            # print(f'Error dict: {error_dict}')
        av_acc = fold_acc / len(folds)
        av_valid = fold_valid / len(folds)
        print(f'Skipped: {total_skipped}')
        print(f'Average accuracy: {av_acc}%')
        print(f'Average not valid: {100-av_valid}%')
        print(f'Total erros in the data: {errors_total}')
        # print(f'Error dict: {error_dict}')
        with open(error_dict_file, 'wb') as f:
            pickle.dump(error_dict, f)
        out_dict = dict()
        # print(total_types_result)
        for key, value in total_types_result.items():
            type_acc = round((value['correct'] * 100) / value['all'])
            out_dict[key] = type_acc

        result = [(k, out_dict[k]) for k in sorted(out_dict, key=out_dict.get, reverse=True)]
        for k, v in result:
            all_num = total_types_result[k]['all']
            print(f'{k}: {v}%, # of errors: {all_num}')


def build_out_table():
    models = [dir for dir in os.listdir(translated) if os.path.isdir(translated + dir)]
    path_to_errors = 'error_dict.pkl'
    out_file = codecs.open('out_moses_table.csv', 'w')
    out_writer = csv.writer(out_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    c = 0
    with open(path_to_errors, 'rb') as pickle_file:
        error_data = pickle.load(pickle_file)
        # print(len(error_data.keys()))
        # print(error_data)
        # header = ['error_id', 'type', 'error_length', 'suggestion', 'is_correct']
        # for model in models:
        #     header.append(model + '_is_suggested')
        #     header.append(model + '_score')
        # out_writer.write(header)

        for key, value in error_data.items():
            c += 1
            row = []
            # print(key)
            # print(value)
            error_info = []
            error = key.split('_')[-1]
            for k, v in value.items():
                if k != 'type':
                    model_dict = v
                    if model_dict['corrected'] == 1:
                        correction = model_dict['correction']
                        score = model_dict['score']
                        corrected = 1

                    elif model_dict['corrected'] == 0 and model_dict['valid'] == 1:
                        correction = None
                        score = 0.0
                        corrected = 0
                    else:
                        correction = model_dict['suggestion']
                        score = model_dict['score']
                        corrected = 0
                    error_info.append((k, correction, score, corrected))
                else:
                    t = v[0]
            print('ERROR INFO', error_info)
            suggestion = None
            for tuple in error_info:
                if tuple[1] != None and suggestion != tuple[1]:
                    suggestion = tuple[1]
                    cor = tuple[3]
                    row = [key, t, len(error), suggestion, cor]
                    for t in error_info:
                        if t[1] == suggestion:
                            row.append(1)
                            row.append(t[2])
                        elif not t[1]:
                            row.append(0)
                            row.append(t[2])
                        else:
                            row.append(-1)
                            row.append(t[2])
                    print(row)
                    out_writer.write(row)
                else:
                    continue

            if c > 10:
                sys.exit()


def main():
    # get_bleu_score()
    #eval()
    build_out_table()


if __name__ == '__main__':
    main()
