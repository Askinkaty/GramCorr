# -*- coding: utf-8 -*-

import os
import codecs
import csv

import Levenshtein
import sys

data_dir = '/Users/katinska/GramCorr/mtensemble/input/new_folds_30082021'
out_data_dir = '/Users/katinska/GramCorr/mtensemble/input/folds_with_spell_30082021'


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


def split_list(all_rows, n):
    all_folds = []
    av_len = len(all_rows) / n
    for i in range(n):
        new_fold = []
        if i != n - 1:
            while len(new_fold) < int(av_len):
                new_fold.append(all_rows[-1])
                all_rows = all_rows[:-1]
            all_folds.append(new_fold)
        else:
            all_folds.append(all_rows)
    return all_folds


def get_hunspell_suggestions():
    suggested = codecs.open(os.path.join(out_data_dir, 'hunspell_corrections.txt'), mode='r')
    filtered_lines = []
    for sline in suggested:
        if sline.strip() is None:
            continue
        else:
            filtered_lines.append(sline)

    suggestion_dict = dict()
    for s in filtered_lines:
        if s.startswith('&'):
            split = s.split(':')
            error = split[0].split(' ')[1].strip()
            ss = split[1].split(',')
            if error not in suggestion_dict:
                suggestion_dict[error] = [(el.strip(), i) for i, el in enumerate(ss)]
    for k, v in suggestion_dict.items():
        suggestion_dict[k] = v[:5]
    return suggestion_dict


# Spell-checker == change to 1 if the line was suggested by SP, not if it was =to expected

def get_suggections():
    suggestion_dict = get_hunspell_suggestions()

    files = os.listdir(data_dir)
    all_rows = []
    header = None
    already_added = []
    c = 0
    new_sug = 0
    for file in files:
        if file.endswith('.csv') and 'fold' in file:
            # file = 'out_moses_table_test.csv'
            # data_dir = '.'
            print(file)
            with codecs.open(os.path.join(data_dir, file), mode='r') as table_file:
                table_reader = csv.reader(table_file, delimiter='\t')
                for i, row in enumerate(list(table_reader)):
                    # if row[0] == 'fold8_3419_ID2285.txt_42_62_62_indenen':
                    #     print('ROW', row)

                    if i == 0:
                        header = row[:-1]
                        header.append('spellcheker_suggested')
                        header.append('spellcheker_score')
                        header.append('spellcheker_rank')
                    else:
                        c += 1
                        # print(row)
                        expected = row[-1]
                        # if row[0] != 'fold8_3419_ID2285.txt_42_62_62_indenen':
                        #     continue
                        # else:
                        #     print(row)
                        error = row[0].split('_')[-1]

                        if error not in suggestion_dict:
                            new_line = row[:-1]
                            new_line.append('0')
                            new_line.append('-100')
                            new_line.append('-1')
                            if new_line not in all_rows:
                                all_rows.append(new_line)
                        else:
                            # print(suggestion_dict[error])
                            # print(already_added)
                            for tpl in suggestion_dict[error]:
                                e = tpl[0]
                                rank = tpl[1]
                                already_suggested = False
                                n = row[0] + '_' + e
                                # print('N', n)
                                with codecs.open(os.path.join(data_dir, file), mode='r') as table_file1:
                                    table_reader_j = csv.reader(table_file1, delimiter='\t')
                                    for j, j_row in enumerate(list(table_reader_j)):
                                        if e == j_row[3] and row[0] == j_row[0]:
                                            already_suggested = True
                                            break
                                # print('ALREDY SUGG', already_suggested, e)
                                distance = Levenshtein.distance(e, error)
                                m = row[0] + '_' + row[3]
                                # print(row[3], e, row[3] == e)
                                # new suggestion

                                # print(already_added)
                                if not already_suggested and n not in already_added and row[3] != e:
                                    # print('Here 1')
                                    new_line = row[:5]
                                    for i in range(4):
                                        new_line.extend([0, 0.0, -1])
                                    new_line[3] = e
                                    if e == expected:
                                        new_line[4] = '1'
                                    else:
                                        new_line[4] = '0'
                                    new_line.append('1')
                                    new_line.append(distance)
                                    new_line.append(rank)
                                    if new_line not in all_rows:
                                        new_sug += 1
                                        all_rows.append(new_line)
                                        already_added.append(n)

                                # we've already recorded this suggestion
                                if not already_suggested and n in already_added and row[3] != e:
                                    continue

                                # error which was suggested by other models
                                # print('Ald sug', already_suggested, n not in already_added, row[3]==e)
                                if already_suggested and n not in already_added and row[3] == e:
                                    # print('Here 2', e)
                                    new_line = row[:-1]
                                    # print(row[3])
                                    if row[3] == e:
                                        new_line.append('1')
                                        new_line.append(distance)
                                        new_line.append(rank)
                                        if new_line not in all_rows:
                                            all_rows.append(new_line)
                                            already_added.append(n)


                                # line with suggestion which was not proposed by spell-checker
                                if m not in already_added and row[3] != e:
                                    # print('Here 3', e)

                                    new_line = row[:-1]
                                    # print('New_line: ', new_line)
                                    new_line.append('0')
                                    new_line.append('-100')
                                    new_line.append('-1')

                                    if new_line not in all_rows:
                                        all_rows.append(new_line)
                                        already_added.append(m)

                                # print('All rows:', all_rows)

                                # if not already_suggested and m not in already_added and row[3] != e:
                                #     new_line = row[:-1]
                                #     # print('New_line: ', new_line)
                                #     new_line.append('0')
                                #     new_line.append('-100')
                                #     new_line.append('-1')
                                #
                                #     if new_line not in all_rows:
                                #         all_rows.append(new_line)
                                #         already_added.append(m)

                                # if already_suggested:
                                #     print('N', n, n in already_added)
                                #     if n not in already_added:
                                #         print('Here:', e)
                                #         new_line = row[:-1]
                                #         print(row[3])
                                #         if row[3] == e:
                                #             new_line.append('1')
                                #             new_line.append(distance)
                                #             new_line.append(rank)
                                #             print('All rows:', all_rows)
                                #             if new_line not in all_rows:
                                #                 all_rows.append(new_line)
                                #                 already_added.append(n)
                                #     else:
                                #         continue
                                # else:
                                #     if n not in already_added:
                                #         new_line = row[:5]
                                #         for i in range(4):
                                #             new_line.extend([0, 0.0, -1])
                                #         new_line[3] = e
                                #         if e == expected:
                                #             new_line[4] = '1'
                                #         else:
                                #             new_line[4] = '0'
                                #         new_line.append('1')
                                #         new_line.append(distance)
                                #         new_line.append(rank)
                                #         if new_line not in all_rows:
                                #             new_sug += 1
                                #             all_rows.append(new_line)
                                #             already_added.append(n)
                                #     else:
                                #         continue

    print(len(all_rows))
    print(c)
    print(new_sug)
    print(len(already_added))
    all_folds = split_list(all_rows, len(files))
    for k, file in enumerate(files):
        if file.endswith('.csv') and 'fold' in file:
            print(file)
            with codecs.open(os.path.join(out_data_dir, file), mode='w') as full_out:
                writer = csv.writer(full_out, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(header)
                for el in all_folds[k]:
                    writer.writerow(el)


import random
import numpy as np


def change_split():
    out_dir = '/Users/katinska/GramCorr/mtensemble/input/folds_with_spell_30082021'

    files = os.listdir(out_data_dir)
    errors_lines = []
    cur_err = None
    er_line = []
    header = None
    for file in files:
        if file.endswith('.csv') and 'fold' in file:
            with codecs.open(os.path.join(out_data_dir, file), mode='r') as table_file:
                table_reader = csv.reader(table_file, delimiter='\t')
                for i, row in enumerate(table_reader):
                    if i == 0:
                        header = row
                    else:
                        if cur_err != row[0]:
                            if cur_err is None:
                                er_line.append(row)
                                cur_err = row[0]
                            else:
                                cur_err = row[0]
                                errors_lines.append(er_line)
                                er_line = []
                                er_line.append(row)
                        else:
                            er_line.append(row)
    print(len(errors_lines))
    rows = []
    for e in errors_lines:
        for r in e:
            rows.append(r)
    print(len(rows))

    # print(errors_lines[0])
    # print(header)
    random.shuffle(errors_lines)
    splits = np.array_split(errors_lines, 10)
    for j, split in enumerate(splits):
        file = 'fold' + str(j) + '.csv'
        with codecs.open(os.path.join(out_dir, file), mode='w') as out:
            writer = csv.writer(out, delimiter='\t')
            writer.writerow(header)
            for er in split:
                for line in er:
                    writer.writerow(line)


def count_error_sugg():
    dir = '/Users/katinska/GramCorr/mtensemble/input/folds_with_spell_042021'

    files = os.listdir(dir)
    files = ['/Users/katinska/GramCorr/evaluation/out_moses_table_042021.csv']
    errors = []
    for file in files:
        # if file.endswith('.csv') and 'fold' in file:
        if file.endswith('.csv'):

            print(file)
            # with codecs.open(os.path.join(out_data_dir, file), mode='r') as table_file:
            with codecs.open(os.path.join(file), mode='r') as table_file:

                table_reader = csv.reader(table_file, delimiter='\t')
                for i, row in enumerate(table_reader):
                    if i == 0:
                        header = row
                    else:
                        errors.append(row[0] + '_' + row[3])
    print(len(errors))


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
    change_split()
    # count_error_sugg()