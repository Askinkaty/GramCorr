# -*- coding: utf-8 -*-

import os
import codecs
import re
import csv

from sklearn.model_selection import train_test_split

"""
Lines with errors in the dataset look like the following one: 
"Jeder hat sicher <error type="09 tog instead of sep: other cases">schonmal //// schon mal</error> aus Gruppenzwang oder weil er es nicht besser wusste etwas Dummes angestellt."
"""
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
    random_split = True
    line_counter = 0
    all_correct_lines = 0
    error_types = dict()

    names = [name for name in os.listdir(corpus_directory) if name.endswith(".txt")]
    error_table = codecs.open(error_table_file, 'a')
    error_count = codecs.open(error_count_file, 'a')
    writer_er_type = csv.writer(error_table)
    writer_er_count = csv.writer(error_count)
    writer_er_count.writerow(['error_type', 'source', 'target'])
    writer_er_type.writerow(['error_type', 'count'])

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
                            error_type = m.group('type').strip()
                            er_line = re.sub(r'<error type.*/error>', error, line)
                            cor_line = re.sub(r'<error type.*/error>', correct, line)
                            error_types[error_type] = error_types.get(error_type, 0) + 1
                            outfile_source.write(er_line)
                            outfile_target.write(cor_line)
                            writer_er_type.writerow([error_type, correct, error])
                    else:
                        all_correct_lines += 1
                        outfile_source.write(line)
                        outfile_target.write(line)
        outfile_source.close()
        outfile_target.close()
    sorted_err_counts = {k: error_types[k] for k in sorted(error_types, key=error_types.get, reverse=True)}
    for k, v in sorted_err_counts.items():
        writer_er_count.writerow([k, v])

    print('# correct lines:', all_correct_lines)
    print('# all lines:', line_counter)
    print('# lines with errors:', line_counter - all_correct_lines)

# number of files in random split
# 961
# 301
# 241