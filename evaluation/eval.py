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

error_table_file = '../corpora/LearnerCorpora/Koko/cv/error_coordinates.csv'
translated_folds = '../translated/moses/Koko/cv/'


def get_bleu_score():
    # Bleu score would be probably very high because sentences are almost the same anyway
    folds = [dir for dir in os.listdir(translated_folds) if os.path.isdir(translated_folds + dir)]
    fbleu = 0
    fgleu = 0
    for fold in folds:
        target_file = os.path.join(translated_folds, fold + '/train.de')
        trans_file = os.path.join(translated_folds, fold + '/train.en.trans.de')
        target = codecs.open(target_file, 'r', encoding='utf-8') # reference
        trans = codecs.open(trans_file, 'r', encoding='utf-8') # correction
        references = []
        hypotheses = []
        for pair in zip(target, trans):
            ref = [pair[0].split()]
            hp = pair[1].split()
            references.append(ref)
            hypotheses.append(hp)
        fbleu += nltk.translate.bleu_score.corpus_bleu(references, hypotheses)
        fgleu += gleu.corpus_gleu(references, hypotheses)
    av_bleu = fbleu / len(folds)
    av_gleu = fgleu / len(folds)
    print(f'Average bleu score: {av_bleu}')
    print(f'Average gleu score: {av_gleu}')


def simple_eval():
    err_table = pd.read_csv(error_table_file)
    err_table['fold#'] = err_table['fold#'].apply(lambda x: int(re.sub(r'fold_', '', x)))
    err_table.fillna(value='')
    folds = [dir for dir in os.listdir(translated_folds) if os.path.isdir(translated_folds + dir)]
    fold_acc = 0
    for fold in folds:
        all_errors = 0
        corrected = 0
        print(fold)
        number = int(re.search(r'.*([0-9]+)', fold).group(1))
        source_file = os.path.join(translated_folds, fold + '/train.en')
        target_file = os.path.join(translated_folds, fold + '/train.de')
        trans_file = os.path.join(translated_folds, fold + '/train.en.trans.de')
        n_best_file = os.path.join(translated_folds, fold + '/train.en.nbest.de')
        # files = [f for f in listdir(fold_path) if not f.startswith('.')]
        source = codecs.open(source_file, 'r', encoding='utf-8') # text with errors
        target = codecs.open(target_file, 'r', encoding='utf-8') # reference
        trans = codecs.open(trans_file, 'r', encoding='utf-8') # correction
        n_best = codecs.open(n_best_file, 'r', encoding='utf-8') # best suggestions of the model
        z = zip(source, target, trans)
        all_lines = 0
        cur_line = 0
        print(type(number))
        print('number', number)
        print(0 in err_table['line'])
        for line in z:
            print('Line: ', cur_line)
            all_lines += 1
            errors = err_table.loc[(err_table['fold#'] == number) & (err_table['line'] == cur_line)]
            cur_line += 1
            if errors.empty:
                continue
            else:
                for i, row in errors.iterrows():
                    all_errors += 1
                    ind = row['indx']
                    error = row['error']
                    correction = row['correction']
                    types = row['types']
                    print(f'Error: {error}')
                    print(f'Correction: {correction}')
                    err_line, corr_line, trans_line = line
                    print(err_line)
                    if ind - 5 > 0 and ind+(len(error))+5 < len(err_line):
                        target_span = err_line[ind-5:ind+len(error)+5]
                    elif ind+(len(error))+5 < len(err_line):
                        target_span = err_line[ind+1:ind+len(error)+5]
                    else:
                        target_span = err_line[ind+1:len(error)]
                    if correction in target_span:
                        corrected += 1
                    print(f'Target span: {target_span}')
                    print(f'Corrected: {corrected}')
        fold_acc += round((corrected * 100)/all_errors) # from all errors, how many were corrected as expected?
        print(f'Accuracy: {fold_acc}')
    av_acc = fold_acc/len(folds)
    print(f'Average accuracy: {av_acc}')


def main():
    #simple_eval()
    get_bleu_score()




if __name__== '__main__':
    main()