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
translated = '../translated/Koko_xml/'
data = '../translate/Koko/word_test'
proc_data_dir = '../translate/Koko/split_processed'
error_dict_file = 'error_dict_check1.pkl'
random.seed(42)


def get_bleu_score():
    """
    :return: Bleu (optionally Gleu) score for the translation.
    """
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
            # print('ERROR', )
            # print('TARGET ', self.target)
            # print('TRANSL ', self.translation)
            self.collect_indices()
            # print('INDICES ', self.indices)
            # print('NEW TRANS ', self.new_translation)
            if len(self.new_translation) == 0:
                # print('2')
                return None
            if not self.indices:
                # print('3')
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
            # print('#' * 100)
            # print('ERRORS', self.errors)
            # print('CORRECTIONS', self.corrections)
            # print('HYPOTHESES', self.hypotheses)
            try:
                assert len(self.errors) == len(self.corrections) == len(self.hypotheses)
            except:
                # print('#' * 100)
                # print('ERRORS', self.errors)
                # print('CORRECTIONS', self.corrections)
                # print('HYPOTHESES', self.hypotheses)
                self.skipped = True
                # print('4')
                return None
        return True

    def collect_indices(self):
        end = 0
        self.new_translation = self.translation.replace('\\\"', '"').replace('\\"', '"').replace('\n', '')
        # print(self.new_translation)
        margin = 0
        not_errors = 0
        for j, s in enumerate(self.source):
            s = s.replace('\n', '').replace('\ "', '"')
            if s == '':
                continue
            if not self.__is_error_line(s):
                not_errors += 1
                s = re.escape(s).replace('"', '\\"')
                found_all = list(re.finditer(s, self.new_translation))
                if found_all:
                    if len(found_all) == 1:
                        start = found_all[0].start()
                        end = found_all[0].end()
                        self.indices.append((start, end))
                    else:
                        for found in found_all:
                            # we need it because we could have the same string repeated many times (e.g. 'und')
                            # so we need to know where to start to search
                            if found.start() >= (end + margin):
                                start = found.start()
                                end = found.end()
                                self.indices.append((start, end))
                                break
            elif self.__is_error_line(s) and not self.__clean_error_line(s):
                self.errors_num += 1
                self.errors.append('')
                self.corrections.append(self.__clean_error_line(self.target[j]))
                if not self.indices:
                    self.indices.append((0, 0))
            else:
                self.errors_num += 1
                error = self.__clean_error_line(self.source[j])
                correct = self.__clean_error_line(self.target[j])
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


def collect_lines(source, target, number, char):
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
            if char:
                s_line = glue_characters(s_line)
                t_line = glue_characters(t_line)
            full_s_line.append(s_line)
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
    if char:
        proc_data_dir += '_char'
    source_file = os.path.join(proc_data_dir, fold + '/train.en')
    target_file = os.path.join(proc_data_dir, fold + '/train.de')
    source = codecs.open(source_file, 'r', encoding='utf-8')  # text with errors
    target = codecs.open(target_file, 'r', encoding='utf-8')  # reference
    d = Data(trans, n_best, source, target)
    return d


def get_other_hypotheses(line, source_line, target_line, n_best, correction, error, types, i, corrected, char):
    hypotheses = dict()
    score = 0
    hyp_num = 5
    for b_line in n_best:
        score_line = b_line.split('|||')
        if int(score_line[0]) + 1 == line:
            if corrected and not score:
                score = float(score_line[-1].strip())
            pair = Pair((source_line, target_line, score_line[1]), char)
            result = pair.get_hypotheses()
            if not result or pair.broken or pair.skipped:
                continue
            else:
                # corrected, not_corrected, new_suggestion, types_result
                n_c, n_v, new_suggestion, _ = check_hypothesis(correction, error, types,
                                                               pair.hypotheses[i])
                if pair.hypotheses[i] not in hypotheses.keys():
                    hypotheses[pair.hypotheses[i]] = float(score_line[-1].strip())
                if len(hypotheses) > hyp_num:
                    break
    return hypotheses, score


def eval():
    models = [dir for dir in os.listdir(translated) if os.path.isdir(translated + dir)]
    err_table = pd.read_csv(error_table_file, delimiter='\t')
    err_table.fillna('', inplace=True)
    error_dict = dict()
    missing = 0
    all_av_not_corrected = 0
    all_av_corrected = 0
    for model in models:
        # if '5' not in model:
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
                pair = Pair(p, char)
                result = pair.get_hypotheses()
                total_skipped += int(pair.skipped)
                # if cur_line > 40:
                    # sys.exit()
                if not result or pair.broken:
                    # print('RESULT:', result)
                    # print('Broken: ', pair.broken)
                    continue
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
                            total_types_result = update_dict(total_types_result, types_result)
                            corrected += int(c)
                            not_corrected += int(v)
                            new_suggestions += int(new_suggestion)
                            # print('Corrected:', c)
                            new_hypotheses, score = get_other_hypotheses(cur_line, p[0], p[1], data.n_best,
                                                                      pair.corrections[l], pair.errors[l], types, l, c,
                                                                          char)
                            error_dict[err_id][model]['suggestions'] = new_hypotheses
                            error_dict[err_id][model]['expected'] = pair.corrections[l]
                            if c:
                                error_dict[err_id][model]['correction'] = pair.corrections[l]
                                error_dict[err_id][model]['score'] = score


            facc = round((corrected * 100) / all_errors)
            fnotcorr = round((not_corrected * 100) / all_errors)
            fsugg = round((new_suggestions * 100)/all_errors)
            fold_acc += facc  # from all errors, how many were corrected as expected?
            fold_not_corrected += fnotcorr
            fold_sugg += fsugg
            errors_total += all_errors
            print(f'Accuracy for {fold}: {facc}%')
            print(f'Average not corrected at all {fold}: {fnotcorr}%')
            print(f'Average new suggestions for {fold}: {fsugg}%')
        av_acc = fold_acc / len(folds)
        av_not_corr = fold_not_corrected / len(folds)
        av_sugg = fold_sugg / len(folds)
        all_av_not_corrected += av_not_corr
        all_av_corrected += av_acc
        print(f'Skipped: {total_skipped}')
        print(f'Average fold accuracy: {av_acc}%')
        print(f'Average fold not corrected at all : {av_not_corr}%')
        print(f'Average fold new suggestions : {av_sugg}%')
        # make_type_report(total_types_result)
    # print(f'Error dict: {error_dict}')
    print(f'All errors #: {errors_total}')
    print(f'Missing errors #: {missing}')
    print(f'All av. not corrected #: {all_av_not_corrected/4}')
    print(f'All av. corrected #: {all_av_corrected/4}')

    with open(error_dict_file, 'wb') as f:
        pickle.dump(error_dict, f)


def collect_error_info(error_id, data, models):
    error_info = []
    model_list = []
    error = error_id.split('_')[-1]
    for k, v in data.items():
        if k != 'type':
            model_dict = v
            expected = model_dict['expected']
            if model_dict['corrected'] == 1:
                correction = model_dict['correction']
                score = model_dict['score']
                corrected = 1
                model_list.append(k)
                error_info.append((expected, k, correction, score, corrected))
            if model_dict['suggestions']:
                for suggestion, s_score in model_dict['suggestions'].items():
                    if suggestion == error:
                        continue
                    new_suggestion = (expected, k, suggestion, s_score, int(suggestion == model_dict['expected']))
                    if new_suggestion not in error_info:
                        error_info.append(new_suggestion)
            if model_dict['valid'] == 1 and not model_dict['suggestions']:
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
                    row.append(-1)
                    other_score = max(other_models[m].values())
                    row.append(other_score)
            rows.append(row)
    return rows


def build_out_table():
    models = [dir for dir in os.listdir(translated) if os.path.isdir(translated + dir)]
    path_to_errors = 'error_dict.pkl'
    out_file = codecs.open('out_moses_table_last.csv', 'w')
    out_writer = csv.writer(out_file, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
    c = 0
    corrected = 0
    with open(path_to_errors, 'rb') as pickle_file:
        error_data = pickle.load(pickle_file)
        print(len(error_data.keys()))
        header = ['error_id', 'type', 'error_length', 'suggestion', 'is_correct']
        for model in models:
            header.append(model + '_is_suggested')
            header.append(model + '_score')
        out_writer.writerow(header)
        all_errors = []
        for key, value in error_data.items():
            c += 1
            error_info, t = collect_error_info(key, value, models)
            corrected += any([el[-1] for el in error_info])
            # print(corrected)
            if not any([el[-1] for el in error_info]):
                continue
            for row in build_rows(error_info, t, key, models):
                if row not in all_errors:
                    all_errors.append(row)
            # if c > 30:
            #     break
                    # sys.exit()
        for a_er in all_errors:
            out_writer.writerow(a_er)
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
    out_dir = '/Users/katinska/GramCorr/mtensemble/input/new_folds_last'
    for i, e in enumerate(folds):
        random.shuffle(e)
        with open(os.path.join(out_dir, 'fold'+str(i)+'.csv'), mode='w') as out_file:
            writer = csv.writer(out_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            header = ['expected', 'error_id', 'type', 'error_length', 'suggestion', 'is_correct']
            for model in models:
                header.append(model + '_is_suggested')
                header.append(model + '_score')
            writer.writerow(header)
            for error in e:
                if not any(isinstance(el, list) for el in error):
                    writer.writerow(error)
                else:
                    for hyp in error:
                        writer.writerow(hyp)


def write_fill_data(full, models):
    out_dir = '/Users/katinska/GramCorr/mtensemble/input/new_folds_last'
    full_data_file = os.path.join(out_dir, 'data.csv')
    with open(full_data_file, mode='w') as full_out:
        random.shuffle(full)
        fwriter = csv.writer(full_out, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        header = ['expected', 'error_id', 'type', 'error_length', 'suggestion', 'is_correct']
        for model in models:
            header.append(model + '_is_suggested')
            header.append(model + '_score')
        fwriter.writerow(header)
        for f_er in full:
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
    with codecs.open('out_moses_table_last.csv') as table_file:
        next(table_file, None)
        table_reader = csv.reader(table_file, delimiter='\t')
        for row in table_reader:
            fold = row[0].split('_')[0]
            if fold not in folds:
                folds[fold] = []
            folds[fold].append(row)
    for k, v in folds.items():
        v = collect_same_er(v)
        random.shuffle(v)
        c = chunks(v, 10)
        for j, chunk in enumerate(c):
            print(len(chunk))
            new_folds[j].extend(chunk.tolist())
            full.extend(chunk.tolist())
    print(len(full))
    write_folds(new_folds, models)
    write_fill_data(full, models)


def main():
    # get_bleu_score()
    eval()
    # all_errors, got_suggections, corrected = build_out_table()
    # print(f'All errors in the data: {all_errors}')
    # print(f'All errors which left after filtering broken: {got_suggections}')
    # print(f'Corrected by at least one system: {corrected}')
    # split_table()


if __name__ == '__main__':
    main()

