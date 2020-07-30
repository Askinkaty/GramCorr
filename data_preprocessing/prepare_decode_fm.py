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
from collections import namedtuple


# Input: A sentence + edit block in an m2 file.
# Output 1: The original sentence (a list of tokens)
# Output 2: A dictionary; key is coder id, value is a tuple.
# tuple[0] is the corrected sentence (a list of tokens), tuple[1] is the edits.
# Process M2 to extract sentences and edits.

def processM2(info):
    info = info.split("\n")
    orig_sent = info[0][2:].split()  # [2:] ignore the leading "S "
    # print(orig_sent)
    # cor_sent = orig_sent[:]
    # gold_edits = []
    # coder = 0
    all_edits = info[1:]
    # print(all_edits)
    # Simplify the edits and group by coder id.
    edit_dict = processEdits(all_edits)
    out_dict = {}
    # Loop through each coder and their edits.
    marked = []
    for coder, edits in edit_dict.items():
        # print(coder)
        # Copy orig_sent. We will apply the edits to it to make cor_sent
        cor_sent = orig_sent[:]
        marked_correct = orig_sent[:]
        gold_edits = []
        offset = 0
        for edit in edits:
            # Do not apply noop or Um edits, but save them
            if edit[2] in {"noop", "Um"}:
                gold_edits.append(edit + [-1, -1])
                continue
            orig_start = edit[0]
            orig_end = edit[1]
            cor_toks = edit[3].split()
            error = orig_sent[orig_start:orig_end]
            if error:
                error = error[0]
                if type(error) is not list:
                    if error not in string.punctuation and not error.isdigit() and error != '...':
                        error = '++' + error + '++'
            orig_sent[orig_start:orig_end] = [error]


            # print('here', orig_sent[orig_start:orig_end])
            # orig_sent[orig_start:orig_end] = '++' + orig_sent[orig_start:orig_end] + '++'

            # Apply the edit.
            cor = []
            if edit[3]:
                for e in edit[3].split():
                    if e not in string.punctuation and not e.isdigit() and e != '...':
                        cor.append('++' + e + '++')
                    else:
                        cor.append(e)
            marked_correct[orig_start + offset:orig_end + offset] = cor
            # print(marked_correct)

            cor_sent[orig_start + offset:orig_end + offset] = cor_toks
            # print(cor_sent)
            # Get the cor token start and end positions in cor_sent
            cor_start = orig_start + offset
            cor_end = cor_start + len(cor_toks)
            # Keep track of how this affects orig edit offsets.
            offset = offset - (orig_end - orig_start) + len(cor_toks)
            # Save the edit with cor_start and cor_end
            gold_edits.append(edit + [cor_start] + [cor_end])
            # marked.append(marked_correct)
        # Save the cor_sent and gold_edits for each annotator in the out_dict.
        # print(cor_sent)
        # print(out_dict)
        # print(gold_edits)
        # print(coder)
    out_dict[coder] = (cor_sent, gold_edits)
    # print(out_dict)
    # print(marked)
    # print(len(marked))
    return orig_sent, out_dict, marked_correct


# Input: A list of edit lines for a sentence in an m2 file.
# Output: An edit dictionary; key is coder id, value is a list of edits.
def processEdits(edits):
    edit_dict = {}
    for edit in edits:
        edit = edit.split("|||")
        span = edit[0][2:].split()  # [2:] ignore the leading "A "
        start = int(span[0])
        end = int(span[1])
        cat = edit[1]
        cor = edit[2]
        id = edit[-1]
        # print(start)
        # print(end)
        # print(cor)
        # Save the useful info as a list
        proc_edit = [start, end, cat, cor]
        # Save the proc edit inside the edit_dict using coder id.
        if id in edit_dict.keys():
            edit_dict[id].append(proc_edit)
        else:
            edit_dict[id] = [proc_edit]
    # print(edit_dict)
    return edit_dict


def mark_errors(out_file, out_target_file, in_file):
    # Setup output m2 file
    out_parallel = codecs.open(out_file, "w")
    out_target = codecs.open(out_target_file, 'w')

    print("Processing files...")
    # Open the m2 file and split into sentence+edit chunks.
    m2_file = codecs.open(in_file).read().strip().split("\n\n")

    for info in m2_file:
        # Get the original and corrected sentence + edits for each annotator.
        orig_sent, coder_dict, marked_correct = processM2(info)
        # print(orig_sent)
        # print(coder_dict)
        # Save info about types of edit groups seen
        # Only process sentences with edits.
        if coder_dict:
            # Save marked up original sentence here, if required.
            proc_orig = ""
            # Loop through the annotators
            for coder, coder_info in sorted(coder_dict.items()):
                cor_sent = coder_info[0]
                # print('ORIGINAL', orig_sent)
                # print('CORRECT', cor_sent)
                # print(type(orig_sent))
                # print(type(cor_sent))
                # out_parallel.write(" ".join(orig_sent) + "\t" + " ".join(cor_sent) + "\n")
                s = " ".join(str(x) for x in orig_sent if type(x) is not list)
                m = " ".join(str(x) for x in marked_correct if type(x) is not list)
                # print('s', s)
                # print(type(s))
                out_parallel.write(s + "\n")
                out_target.write(m + "\n")

    out_parallel.close()

def split_character(line):
    """
    :param line: tokenized line
    :return:  character split line
    """
    line = line.replace(' ', '_')
    ch_line = ' '.join(list(line)).strip()
    return ch_line

def build_line(line, char):
    line = line.replace('\ "', '\\"').replace('"', '\\"').replace('<', '').replace('>', '')
    line = line.rstrip('\n')
    words = line.split(' ')
    new_words = []
    target_ids = []
    for i, word in enumerate(words):
        if '++' in word:
            target_ids.append(i)
    good_list = []
    for j, word in enumerate(words):
        if j not in target_ids:
            good_list.append(word)
            if j == len(words) - 1:
                good_string = ' '.join(good_list)
                if len(new_words):
                    if char:
                        good_string = '<np translation="' + '_' + split_character(good_string) + '_' + '">' + '_' +\
                                      split_character(good_string) + '_' + '</np>'
                    else:
                        good_string = '<np translation="' + ' ' + good_string + ' ' + '">' +\
                                      ' ' + good_string + ' ' + '</np>'

                else:
                    if char:
                        good_string = '<np translation="' + split_character(good_string) + '_' + '">' +\
                                      split_character(good_string) + '_' + '</np>'
                    else:
                        good_string = '<np translation="' + good_string + ' ' + '">' + good_string + ' ' + '</np>'
                new_words.append(good_string)
        else:
            good_string = ' '.join(good_list)
            if len(new_words):
                if char:
                    good_string = '<np translation="' + '_' + split_character(good_string) + '_' + '">' + '_' +\
                                  split_character(good_string) + '_' + '</np>'
                else:
                    good_string = '<np translation="' + ' ' + good_string + ' ' + '">' + ' ' + good_string + ' ' + '</np>'

            else:
                if char:
                    good_string = '<np translation="' + split_character(good_string) + '_' + '">' + \
                                  split_character(good_string) + '_' + '</np>'
                else:
                    good_string = '<np translation="' + good_string + ' ' + '">' + good_string + ' ' + '</np>'
            new_words.append(good_string)
            good_list = []
            if char:
                new_words.append(split_character(word.replace('++', '')))
            else:
                new_words.append(word.replace('++', ''))
    result = ''.join(new_words)
    return result


def prepare_for_decode(in_file, out_file):
    char = False
    if char:
        out = codecs.open(out_file + '_char', 'w')
    else:
        out = codecs.open(out_file, 'w')
    inp = codecs.open(in_file).readlines()
    for i, line in enumerate(inp):
        new_line = build_line(line, char)
        out.write(new_line + "\n")
    out.close()


if __name__ == '__main__':
    data_path = '/Users/katinska/GramCorr/corpora/LearnerCorpora/Merlin_Falko/data'
    splits = ['test', 'dev', 'train']
    files = [f for f in os.listdir(data_path)]
    # for split in splits:
    #     for f in files:
    #         if 'm2' in f and split in f:
    #             m2file = os.path.join(data_path, f)
    #             out_file = os.path.join(data_path, 'tagged_' + split + '.src')
    #             out_target = os.path.join(data_path, 'tagged_' + split + '.trg')
    #             mark_errors(out_file, out_target, m2file)

    # # DE
    for split in splits:
        in_file = '/Users/katinska/GramCorr/corpora/LearnerCorpora/Merlin_Falko/data/tagged_' + split + '.src'
        out_file = '/Users/katinska/GramCorr/corpora/LearnerCorpora/Merlin_Falko/data/decode_' + split + '.src'
        prepare_for_decode(in_file, out_file)
    # #RU
    in_file = '/Users/katinska/GramCorr/Rus/snippets_translit/test/train.en'
    out_file = '/Users/katinska/GramCorr/Rus/decode_test_train.en'
    prepare_for_decode(in_file, out_file)