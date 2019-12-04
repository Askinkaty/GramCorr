# -*- coding: utf-8 -*-

import os
import codecs
import re
import csv
import sys
import zipfile
import argparse
import random
import time
import pickle
import string

from sklearn.model_selection import train_test_split

"""
Lines with errors in the dataset look like the following one: 
"Jeder hat sicher <error type="09 tog instead of sep: other cases">schonmal //// schon mal</error> aus Gruppenzwang oder weil er es nicht besser wusste etwas Dummes angestellt."


"""

random.seed(42)


# You need to have a Koko downloaded
corpus_directory = "../corpora/LearnerCorpora/Koko/"
corpus_file = corpus_directory + "Koko.zip"
original_split_ind = [1089, 1663, 2225]

error_table_file = 'errors_source_target.csv'
error_count_file = 'error_statistics.csv'
error_coordinates_file = 'error_coordinates.csv'


def get_original_split(names):
    """
    :param names: list of filename in the dir
    :return: list of lists with split names
    """
    split_names = []
    split = []
    for name in names:
        if not any([str(id) in name for id in original_split_ind]) and not name == names[-1]:
            split.append(name)
        else:
            if name == names[-1]:
                split.append(name)
            split_names.append(split)
            split = []
    return split_names


def split_train_dev_test(names):
    """
    :param names: list with names of files
    :return: 80% of data - train set, 10% - dev, 10% - test
    """
    train, test = train_test_split(names, test_size=0.2, random_state=42)
    test, valid = train_test_split(test, test_size=0.5, random_state=42)
    return [train, test, valid]


def split_cross_val(names, n):
    """
    :param names: list of files to split
    :param n: number of folds to split into
    :return:
    """
    random.shuffle(names)
    k, m = divmod(len(names), n)
    return list(names[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def clean_line(line, char):
    if char:
        line = line.replace('-unreadable-', '@')
    else:
        line = line.replace('-unreadable-', 'UNK')
    # we need to replace error xml tags with $ because we will remove all <> symbols from text
    # sinse they break moses partial translation
    # we need to do it now because we will loose error indexes otherwise
    line = line.replace('"', '\\"').replace('<error', '$error').replace('">', '"$').replace('</error>', '$/error$')
    line = line.replace('>', '').replace('<', '')
    # line = line.replace('>', '\>').replace('<', '\<').replace('"', '\\"')
    line = ' '.join(line.split())
    return line


def clean_err(er):
    new_er = er.replace('„', '').replace('“', '')\
        .replace('\\"', '').replace('"', '').\
        replace('´', "'").replace('\%', ' \%').\
        replace('[', '').replace(']', '').replace('°', '').\
        replace('  ', '').strip()
    new_er = re.sub(r'\s*-\s*', '-', new_er)
    return new_er


def find_error(line):
    """
    :param line: raw line with annotation
    :return: indexes to be replaced, error correction; found error types and info if the line is broken
    """
    to_replace = []
    found_types = dict()
    global number_of_errors
    if '$error type' in line:
        # m = re.finditer(r'<error type="(?P<type>.*?)".*?>(?P<error>.*?)\/{4}(?P<correct>.*?)<\/error>', line)
        m = re.finditer(r'\$error type=\\"(?P<type>.*?)\\".*?\$(?P<error>.*?)\/{4}(?P<correct>.*?)\$\/error\$', line)
        if m:
            for match in m:
                ind = [match.start(), match.end()]
                # print(match)
                # sys.exit()
                i = ind[0]
                k = ind[1]
                error = match.group('error').strip()
                number_of_errors += 1
                correct = match.group('correct').strip()
                broken = check_correction(error)
                error_info = match.group('type').strip()
                if '//' in error_info:
                    types = [el.strip() for el in error_info.split('//')]
                else:
                    types = [error_info]
                for t in types:
                    if t not in found_types:
                        found_types[t] = []
                        found_types[t].append([error, correct])
                    else:
                        found_types[t].append([error, correct])
                to_replace.append([ind, error, correct, types])
    # print(to_replace)
    return to_replace, found_types, broken


def split_data(split, names, char, folds=None):
    out_names = []
    if split == 'random':
        split_names = split_train_dev_test(names)  # train, test, valid
        out_dir = corpus_directory + 'random_spit'
    elif split == 'random-cv':
        split_names = split_cross_val(names, folds)
        out_dir = corpus_directory + 'cv'
    elif split == 'original':
        # original split which we could use only for comparison with the prev experiments
        split_names = get_original_split(names)
        out_dir = corpus_directory + 'original_split'
    else:
        split_names = [names]
        out_dir = corpus_directory + 'no_split'
    if char:
        out_dir += '_char'
    for i in range(len(split_names)):
        out_names.append('fold'+ str(i))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    split_names = list(split_names)
    return out_names, split_names, out_dir


def split_character(line):
    """
    :param line: tokenized line
    :return:  character split line
    """
    line = line.replace(' ', '_')
    ch_line = ' '.join(list(line)).strip()
    return ch_line


def split_bpe(line):
    """
    TODO: implement if needed, should use BPE model trained on the data
    :param line: tokenised line
    :return: line split into subwords using BPE
    """
    return True


def split_long_line(line, start):
    m = 50
    parts = []
    if len(line) > m:
        n = round(len(line) / m)
        if len(line) % m:
            n += 1
        s = start
        for j in range(n):
            e = s + m
            piece = line[s:e]
            parts.append(piece)
            s = e
    else:
        parts.append(line)
    return parts


def replace_in_line(line, to_replace):
    """
    :param line: raw line
    :param to_replace: indexes to replace, error and correction
    :return: line with error, corrected line, index of correction, index of error, line with xml mark-up for decoder
    """
    er_line = ''
    cor_line = ''
    new_start = 0
    real_ind = []
    err_ind = []
    dif = 0
    dif_er = 0
    to_preproc_source = []
    to_preproc_target = []

    for i, repl in enumerate(to_replace):
        ind, error, correct, _ = repl
        if i == 0:
            if error == '' and ind[0] > 0:
                ind[0] = ind[0] - 1
            real_ind.append(ind[0])
            err_ind.append(ind[0])
        else:
            if error == '' and ind[0] > 0:
                ind[0] = ind[0] - 1
            ri = ind[0] - dif + 1
            err_i = ind[0] - dif_er + 1
            real_ind.append(ri)
            err_ind.append(err_i)
        if 'UNK' in error or 'UNK' in correct: # remove for now words which have a tag -unreadable- inside them
            print('UNK tag in the source or the target!')
            return None
        start = ind[0]
        end = ind[1]
        d = end - (start + len(correct))
        d_err = end - (start + len(error))
        dif += d
        dif_er += d_err
        er_line += line[new_start:start] + error
        cor_line += line[new_start:start] + correct

        to_preproc_source.append(line[new_start:start])
        to_preproc_target.append(line[new_start:start])
        to_preproc_source.append('$$$ ' + error)
        to_preproc_target.append('$$$ ' + correct)

        new_start = end
    if new_start < len(line):
        er_line += line[new_start:]
        cor_line += line[new_start:]

        to_preproc_source.append(line[new_start:])
        to_preproc_target.append(line[new_start:])

    er_line = ' '.join(er_line.split())
    cor_line = ' '.join(cor_line.split())

    return [er_line, cor_line, real_ind, err_ind, to_preproc_source, to_preproc_target]


def balanced(expression):
    opening = tuple('({[„')
    closing = tuple(')}]“')
    mapping = dict(zip(opening, closing))
    queue = []
    for letter in expression:
        if letter in opening:
            queue.append(mapping[letter])
        elif letter in closing:

            if not queue or letter != queue.pop():
                return False
        else:
            continue
    return not queue


def check_correction(correct):
    global broken_annotations
    symbols = ["{", "}", "(", ")", "[", "]", "„", "“"]
    found = [c for c in correct if c in symbols]
    if len(found) == 1:
        broken_annotations += 1
        return True


def get_err_type_number(name, line):
    try:
        error_number = name.split()[0]
        error_type = name[len(error_number):]
    except:
        print(f'Missing error type in line: {line}')
        error_number = '100'
        error_type = 'unknown'
    return error_type, error_number


def create_csv_tables(files, out_dir):
    """
    :param files: Files which will be created: csv tables with statistics about errors
    :param out_dir: Directory where to write created files
    :return:
    """
    error_table_file, error_count_file, error_coordinates_file = [i for i in files]
    csv_exist = all([os.path.isfile(error_table_file), os.path.isfile(error_count_file)])
    answer = None
    if csv_exist:
        print('csv files with already exist, should I rewrite them? (yes/no)')
        answer = input()
    if not csv_exist or answer == 'yes':
        if answer == 'yes':
            try:
                os.remove(error_table_file)
                os.remove(error_count_file)
                os.remove(os.path.join(out_dir, error_coordinates_file))
            except:
                print("Error while deleting file ")
        error_table = codecs.open(error_table_file, 'a')
        error_count = codecs.open(error_count_file, 'a')
        error_coordinates = codecs.open(os.path.join(out_dir, error_coordinates_file), 'a')
        writer = True
        writer_er_type = csv.writer(error_table)
        writer_er_count = csv.writer(error_count)
        writer_er_coord = csv.writer(error_coordinates, delimiter='\t')
        writer_er_count.writerow(['error_category', 'error_type', 'count'])
        writer_er_type.writerow(['error_category', 'error_type', 'source', 'target', ])
        writer_er_coord.writerow(['fold#', 'line', 'filename', 'origline', 'indx', 'err_indx',
                                  'error', 'correction', 'types'])
        return [writer_er_type, writer_er_count, writer_er_coord, writer]
    else:
        return None


def main():
    args = parser.parse_args()
    split = args.split
    char = args.char
    folds = args.folds

    global number_of_errors
    global broken_annotations
    number_of_errors = 0
    files_with_broken = []
    error_types = dict()
    error_num = dict()
    broken_annotations = 0
    line_counter = 0
    all_correct_lines = 0
    writer = None
    try:
        os.makedirs(corpus_directory)
    except FileExistsError:
        pass
    with zipfile.ZipFile(corpus_file, 'r') as zip_ref:
        zip_ref.extractall(corpus_directory)

    names = [name for name in os.listdir(corpus_directory) if name.endswith(".txt")]
    out_names, split_names, out_dir = split_data(split, names, char, folds)
    preproc_dir = out_dir + '/to_preproc'
    try:
        os.makedirs(preproc_dir)
    except FileExistsError:
        pass

    with open(os.path.join(out_dir, 'split_names.pkl'), 'wb') as pkl:
        pickle.dump(split_names, pkl)
    writers = create_csv_tables([error_table_file,
                                 error_count_file,
                                 error_coordinates_file], out_dir)
    if writers:
        writer_er_type, writer_er_count, writer_er_coord, writer = writers
    for c, subset in enumerate(split_names):

        outfile_source = codecs.open(os.path.join(out_dir, out_names[c] + '_source.txt'), 'w', encoding='utf-8')
        outfile_target = codecs.open(os.path.join(out_dir, out_names[c] + '_target.txt'), 'w', encoding='utf-8')

        out_preproc_source = codecs.open(os.path.join(preproc_dir, out_names[c] + '_source.txt'), 'w', encoding='utf-8')
        out_preproc_target = codecs.open(os.path.join(preproc_dir, out_names[c] + '_target.txt'), 'w', encoding='utf-8')

        fold_line = 0
        for filename in subset:
            with codecs.open(os.path.join(corpus_directory, filename), 'r', encoding='utf-8') as f:
                original_line = 0
                for line in f:
                    original_line += 1
                    fold_line += 1
                    line_counter += 1
                    line = clean_line(line, char)

                    if '$error type' in line:
                        to_replace, found_types, broken = find_error(line)
                        if broken:
                            files_with_broken.append(filename)
                        for name in found_types.keys():
                            error_type, error_number = get_err_type_number(name, line)
                            error_types[error_type] = error_types.get(error_type, 0) + 1
                            error_num[error_type] = error_number
                            if writer:
                                for value in found_types[name]:
                                    writer_er_type.writerow([error_number, error_type, value[0], value[1]])
                        replaced = replace_in_line(line, to_replace)
                        if replaced and to_replace:
                            er_line, cor_line, real_ind, err_ind, \
                            to_preproc_source, to_preproc_target = replaced
                            for ei, er in enumerate(to_replace):
                                new_er = clean_err(er[1])
                                coordinates = [out_names[c], fold_line, filename,
                                               original_line, real_ind[ei], err_ind[ei], new_er, er[2], er[3]]
                                if writer:
                                    writer_er_coord.writerow(coordinates)
                        else:
                            fold_line -= 1
                            original_line -= 1
                            continue
                        er_line = er_line.replace('\ <', '').replace('\ >', '')
                        cor_line = cor_line.replace('\ <', '').replace('\ >', '')
                        if char:
                            er_line = split_character(er_line)
                            cor_line = split_character(cor_line)
                        for sl in to_preproc_source:
                            out_preproc_source.write(sl + '\n')
                        out_preproc_source.write('###' + '\n')
                        for tl in to_preproc_target:
                            out_preproc_target.write(tl + '\n')
                        out_preproc_target.write('###' + '\n')

                        outfile_source.write(er_line + '\n')
                        outfile_target.write(cor_line + '\n')
                    else:
                        all_correct_lines += 1
                        if char:
                            line = split_character(line)
                        line = line.strip()
                        outfile_source.write(line + '\n')
                        outfile_target.write(line + '\n')

                        out_preproc_source.write(line + '\n')
                        out_preproc_target.write(line + '\n')
                        out_preproc_source.write('###' + '\n')
                        out_preproc_target.write('###' + '\n')

        outfile_source.close()
        outfile_target.close()
    sorted_err_counts = {k: error_types[k] for k in sorted(error_types, key=error_types.get, reverse=True)}

    if writer:
        for k, v in sorted_err_counts.items():
            writer_er_count.writerow([error_num[k], k, v])

    for f in names:
        os.remove(os.path.join(corpus_directory, f))

    print('# correct lines:', all_correct_lines)
    print('# all lines:', line_counter)
    print('# lines with errors:', line_counter - all_correct_lines)
    print('Number of errors:', number_of_errors)
    print('Broken annotations:', broken_annotations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert corpus with annotated errors into 1) paralles source/target files line separated",
        formatter_class=argparse.RawTextHelpFormatter,
        usage="%(prog)s [-h] [options] -split TYPE_OF_SPLIT")
    parser.add_argument("-split", choices=["original", "random-cv", "random-3", "none"], default="original",
                        help="Options: original (keep the original split of KOKO), "
                        "random or none", required=True)
    parser.add_argument("-folds", type=int, default=10, help="Number of folds for cross-validation")
    parser.add_argument("-char", action='store_true',
                        help="Keep word tokenised split word in characters: true/false")
    main()
