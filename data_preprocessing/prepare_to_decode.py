# -*- coding: utf-8 -*-
import os
import codecs
import pandas as pd
import re
import sys

#data_dir = '../translate/Koko/folds'
data_dir = '../translate/Koko/split_processed_char'

out_xml_dir = '../translate/Koko_xml/char_test'
out_dir = '../translate/Koko/char_test'


def build_line(line, xml=False):
    # print(line)
    new_line = ''
    for i, el in enumerate(line):
        el = el.replace('\ "', '\\"').replace('"', '\\"').replace('<', '').replace('>', '')
        if el.startswith('$ $ $'):
            el = el.replace('$ $ $', '')
            if xml:
                el = el.strip()
        else:
            el = el.rstrip('\n')
            el += ' '
            if xml:
                if i != 0:
                    el = ' ' + el
                el = '<np translation="' + el + '">' + el + '</np>'
        new_line += el
    new_line = ' '.join(new_line.split())
    new_line += '\n'
    return new_line


def add_decoder_xml_markup():
    folds = [name for name in os.listdir(data_dir)]
    for fold in folds:
        try:
            os.makedirs(os.path.join(out_xml_dir, fold))
        except FileExistsError:
            pass
        try:
            os.makedirs(os.path.join(out_dir, fold))
        except FileExistsError:
            pass
        source_file = codecs.open(os.path.join(data_dir, fold + '/train.en'), 'r', encoding='utf-8')
        target_file = codecs.open(os.path.join(data_dir, fold + '/train.de'), 'r', encoding='utf-8')
        out_source = codecs.open(os.path.join(out_dir, fold + '/train.en'), 'w', encoding='utf-8')
        out_target = codecs.open(os.path.join(out_dir, fold + '/train.de'), 'w', encoding='utf-8')
        out_xml = codecs.open(os.path.join(out_xml_dir, fold + '/train.en'), 'w', encoding='utf-8')
        z = zip(source_file, target_file)
        cor_line_elements = []
        er_line_elements = []
        source_lines = []
        target_lines = []

        for i, line in enumerate(z):
            # if i > 50:
            #     break
            source_line = line[0]
            target_line = line[1]

            if '# # #' not in source_line:
                cor_line_elements.append(target_line)
                er_line_elements.append(source_line)
            else:
                source_lines.append(er_line_elements)
                target_lines.append(cor_line_elements)
                cor_line_elements = []
                er_line_elements = []

        for sl in source_lines:
            new_sline = build_line(sl)
            new_xline = build_line(sl, xml=True)
            # print('ERROR LINE: ', new_sline)
            # print('XML LINE: ', new_xline)
            out_source.write(new_sline)
            out_xml.write(new_xline)
        for tl in target_lines:
            new_tline = build_line(tl)

            # print('CORRECT LINE: ', new_tline)
            out_target.write(new_tline)


if __name__ == '__main__':
    add_decoder_xml_markup()