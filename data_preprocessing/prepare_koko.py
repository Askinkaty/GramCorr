# -*- coding: utf-8 -*-

import os
import codecs
import re
import csv
import sys
import zipfile

from sklearn.model_selection import train_test_split

"""
Lines with errors in the dataset look like the following one: 
"Jeder hat sicher <error type="09 tog instead of sep: other cases">schonmal //// schon mal</error> aus Gruppenzwang oder weil er es nicht besser wusste etwas Dummes angestellt."
"""
corpus_file = "../corpora/LearnerCorpora/Koko/ID2988.zip"
corpus_directory = "../corpora/LearnerCorpora/Koko"

random_split_dir = "../corpora/LearnerCorpora/Koko/random_spit"
original_split_dir = "../corpora/LearnerCorpora/Koko/original_split"
original_split_ind = [1663, 2225]

error_table_file = 'errors_source_target.csv'
error_count_file = 'error_statistics.csv'



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


def split_random(names, val=False):
    valid = None
    train, test = train_test_split(names, test_size=0.2, random_state=42)
    if val:
        train, valid = train_test_split(train, test_size=0.2, random_state=42)
    return [train, test, valid]


if __name__ == "__main__":
    random_split = False
    line_counter = 0
    all_correct_lines = 0
    error_types = dict()
    error_num = dict()

    with zipfile.ZipFile(corpus_file, 'r') as zip_ref:
       zip_ref.extractall(corpus_directory)
    names = [name for name in os.listdir(corpus_directory) if name.endswith(".txt")]
    error_table = codecs.open(error_table_file, 'a')
    error_count = codecs.open(error_count_file, 'a')
    writer_er_type = csv.writer(error_table)
    writer_er_count = csv.writer(error_count)
    writer_er_count.writerow(['error_number', 'error_type', 'count'])
    writer_er_type.writerow(['error_number', 'error_type', 'source', 'target'])


    if random_split:
        split_names = split_random(names, val=True)  # train, test, valid (which is optional)
        outdir = random_split_dir
        out_names = ['train', 'test', 'valid']
    else:
        # original split which we could use only for comparison with the prev experiments
        split_names = get_original_split(names)
        outdir = original_split_dir
        out_names = ['fold1', 'fold2', 'fold3']
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for c, subset in enumerate(split_names):
        outfile_source = codecs.open(os.path.join(outdir, out_names[c] + '_source.txt'), 'a', encoding='utf-8')
        outfile_target = codecs.open(os.path.join(outdir, out_names[c] + '_target.txt'), 'a', encoding='utf-8')
        for filename in subset:
            with codecs.open(os.path.join(corpus_directory, filename), 'r', encoding='utf-8') as f:
                for line in f:
                    line_counter += 1
                    if '<error type' in line:
                        m = re.search(r'<.*type="(?P<type>.*)".*>(?P<error>.*)\/{4}(?P<correct>.*)<', line)
                        if m:
                            error = m.group('error').strip()
                            correct = m.group('correct').strip()
                            if 'unreadable' in error:
                                continue
                            error_info = m.group('type').strip()
                            if '//' in error_info:
                                type_list = [el.strip() for el in error_info.split('//')]
                            else:
                                type_list = [error_info]
                            for el in type_list:
                                try:
                                    error_number = el.split()[0]
                                    error_type = el[len(error_number):]
                                except:
                                    print('Missing error type!')
                                    error_number = '100'
                                    error_type = 'unknown'
                                error_types[error_type] = error_types.get(error_type, 0) + 1
                                error_num[error_type] = error_number
                                writer_er_type.writerow([error_number, error_type, error, correct])
                            er_line = re.sub(r'<error type.*/error>', error, line)
                            cor_line = re.sub(r'<error type.*/error>', correct, line)
                            outfile_source.write(er_line)
                            outfile_target.write(cor_line)
                    else:
                        all_correct_lines += 1
                        outfile_source.write(line)
                        outfile_target.write(line)
        outfile_source.close()
        outfile_target.close()
    sorted_err_counts = {k: error_types[k] for k in sorted(error_types, key=error_types.get, reverse=True)}
    for k, v in sorted_err_counts.items():
        writer_er_count.writerow([error_num[k], k, v])

    print('# correct lines:', all_correct_lines)
    print('# all lines:', line_counter)
    print('# lines with errors:', line_counter - all_correct_lines)

# number of files in random split
# 961
# 301
# 241

