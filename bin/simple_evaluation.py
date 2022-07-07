from evalclass.evaluation import Load
from evalclass.evaluation import Simple_Evaluation
import pandas as pd
import os
import util.file_util as ft
import argparse
import nltk

#nltk.download('treebank')

def args():
    parser = argparse.ArgumentParser(usage='usage', description='Usage of parameters ')
    parser.add_argument('--input', required=False, default='../data/input/kovi/20220427', help='Input path')
    parser.add_argument('--extension', required=False, default='csv', help='Input extension')
    parser.add_argument('--src_lang', required=False, default='ko', help='Source Language')
    parser.add_argument('--tgt_lang', required=False, default='vi', help='Target Language')
    parser.add_argument('--col_names', required=False, default='google', help='Sep: |')
    parser.add_argument('--method', required=False, default='a', help='Select Machine Evaluation Methods'
                                                                      'a = All Method'
                                                                      'b = BLEU'
                                                                      'u = BLEU1'
                                                                      'r = Rouge')
    parser.add_argument('--pos', required=False, default='n', help='Korean POS')
    parser.add_argument('--output', required=False, default='../data/output')
    return parser.parse_args()


def main():
    config = args()

    input_path = config.input
    output = config.output

    src_lang = config.src_lang
    tgt_lang = config.tgt_lang
    extension = config.extension
    col_names = config.col_names
    method = config.method
    pos = config.pos

    flist = ft.get_all_files_sublist(input_path, extension=extension)
    print(flist)

    col_list = [col_names]
    col_list = [c for c in col_list[0].split(',')]

    for fpath in flist:
        print(os.path.splitext(fpath)[1][1:])

        load = Load(input_path)
        evaluation = Simple_Evaluation(src_lang, tgt_lang, col_names, method, pos)

        load.read(fpath)
        print(load.df.head())
        print('Evaluation Data File Length : {}'.format(len(load.df)))
        result_df = evaluation.scoring(load.df)
        summary = evaluation.make_summary(result_df)
        summary.index = col_list

        result_df.to_excel(output + '/' + ft.get_file_name_without_extension(fpath) + '_evaluation.xlsx',
                           index=False)
        summary.to_excel(output + '/' + ft.get_file_name_without_extension(fpath) + '_summary.xlsx')


if __name__ == '__main__':
    main()




