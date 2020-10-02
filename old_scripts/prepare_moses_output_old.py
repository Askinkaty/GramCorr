# -*- coding: utf-8 -*-

import os
import csv
import codecs
import pandas as pd
import re
import sys


translated_dir = '../translated/Koko/'
out_table_file = 'results.csv'
error_table_file = '../corpora/LearnerCorpora/Koko/cv/error_coordinates.csv'
folds = 1


def check_if_corrected(ind, correction, error, trans_line):
    n = 20
    valid = None
    corrected = None

    #
    # print('LINE LEN: ', len(trans_line))
    # print('LINE: ', trans_line)

    if ind - n > 0 and ind + (len(correction)) + n < len(trans_line):
        target_span = trans_line[ind - n:ind + len(correction) + n]
    elif ind - n > 0:
        target_span = trans_line[ind - n:len(trans_line)]
    elif ind + (len(correction)) + n < len(trans_line):
        target_span = trans_line[:ind + len(correction) + n]
    else:
        target_span = trans_line[ind:ind + len(correction)]
    # print('SPAN: ', target_span)
    if correction in target_span:
        if error != '' and error not in target_span:
            corrected = True
            valid = True
    elif error in target_span:
        valid = True
    return corrected, valid, target_span


if __name__ == '__main__':
    models = [dir for dir in os.listdir(translated_dir) if os.path.isdir(translated_dir + dir)]
    err_table = pd.read_csv(error_table_file)
    err_table['fold#'] = err_table['fold#'].apply(lambda x: int(re.sub(r'fold_', '', x)))
    err_table.fillna('', inplace=True)
    out_table = codecs.open(out_table_file, 'w')
    writer = csv.writer(out_table, delimiter='\t')
    header = ['errorId', 'errorType', 'correction span', 'corrected']
    for model in models:
        header.append(model+'_corrected')
        header.append(model+'_score')
    writer.writerow(header)
    error_dict = dict()

    for i, row in err_table.iterrows():
        if i > 30:
            print(error_dict)
            print(len(error_dict))
            sys.exit(0)
        err_id = '_'.join([str(j) for j in row[:5]])
        print(err_id)
        err_type = row['types']
        error_dict[err_id] = dict()
        error_dict[err_id]['type'] = err_type
        for model in models:
            error_dict[err_id][model] = dict()
            model_dir = os.path.join(translated_dir, model)
            for i in range(folds):
                s = [model_dir, 'translated', 'fold'+str(i)]
                #fold_dir = os.path.join(model_dir, 'translated' + str(i))
                #translated_fold_dir = os.path.join(fold_dir, 'translated' + str(i))
                translated_fold_dir = os.path.join(*s)
                print(translated_fold_dir)
                trans_file = os.path.join(translated_fold_dir, 'train.en.trans.de')
                n_best_file = os.path.join(translated_fold_dir, 'train.en.nbest.de')
                trans = codecs.open(trans_file, 'r', encoding='utf-8') # correction
                n_best = codecs.open(n_best_file, 'r', encoding='utf-8')
                t_line = 1
                if i == row['fold#']:
                    corrected = None
                    valid = None
                    for line in trans:
                        if t_line == row['line']:
                            correct = row['correction']
                            error = row['error']
                            ind = row['indx']
                            print(row)
                            print(line)
                            corrected, valid, span = check_if_corrected(ind, correct, error, line)
                            error_dict[err_id][model]['target span'] = span
                            error_dict[err_id][model]['corrected'] = corrected
                            error_dict[err_id][model]['valid'] = valid


                            print('Corrected: ', corrected)
                            print('Valid: ', valid)
                            for b_line in n_best:
                                score_line = b_line.split('|||')
                                if int(score_line[0]) + 1 == t_line:
                                    if corrected:
                                        error_dict[err_id][model]['score'] = float(score_line[-1].strip())
                                        error_dict[err_id][model]['suggestion'] = span
                                        error_dict[err_id][model]['correction'] = correct
                                        break
                                    else:
                                        # if valid:
                                        #     # error_dict[err_id][model]['suggection'] = None
                                        # else:
                                        bcorrected, bvalid, bspan = check_if_corrected(ind, correct, error, score_line[1])
                                        if not bcorrected and bvalid:
                                            #print(bspan)
                                            # print('!!!!!!!!!! Model did not correct this sentence')
                                            continue

                                        else:
                                            print(bspan)
                                            error_dict[err_id][model]['suggestion'] = span
                                            error_dict[err_id][model]['score'] = float(score_line[-1].strip())
                                            print('########## Other model suggestion')
                                            # print(error_dict)
                                            #sys.exit(0)
                                            break

                        t_line += 1
    print(error_dict)
    print(len(error_dict))

    print('__________________________')








