import util.file_util as ft
import util.config_loader as cl
from evalclass.evaluation import Load
from evalclass.evaluation import Evaluation
import pandas as pd
import os
import argparse


def define_arg():
    parser = argparse.ArgumentParser(usage='usage', description='Usage of parameters ')
    parser.add_argument('--config', required=True, help='Configuration ')
    return parser.parse_args()


def main():
    config = cl.ConfigMan('../config/evaluation_config.json')

    input_path = config.get('DATA', 'INPUT_PATH')
    output_path = config.get('DATA', 'OUTPUT_PATH')
    lp_path = config.get('LP', 'LP_PATH')
    ko_character = config.get('KO_CHARACTER', 'KO_CHARACTER')
    extension = config.get('EXTENSION', 'INPUT_EXTENSION')

    lp = pd.read_csv(lp_path, sep='\t')
    flist = ft.get_all_files_sublist(input_path, extension=extension)
    print(flist)

    for i, fpath in zip(range(len(lp)), flist):
        print(os.path.splitext(fpath)[1][1:])

        load = Load(input_path)
        evaluation = Evaluation(lp['src'][i], lp['tgt'][i], lp['place'][i], lp['method'][i], ko_character)

        load.read(fpath)
        print('Evaluation Data File Length : {}'.format(len(load.df)))
        result_index = list(load.df.columns[lp['place'][i]:])
        print(result_index)

        result_df = evaluation.scoring(load.df)

        summary = evaluation.make_summary(result_df)
        print(summary)
        summary.index = result_index

        if ko_character == 'T':
            result_df.to_excel(output_path + '/' + ft.get_file_name_without_extension(fpath) + '_evaluation_M.xlsx',index=False)
            summary.to_excel(output_path + '/' + ft.get_file_name_without_extension(fpath) + '_summary_M.xlsx')
        elif ko_character == 'F':
            result_df.to_excel(output_path + '/' + ft.get_file_name_without_extension(fpath) + '_evaluation.xlsx',index=False)
            summary.to_excel(output_path + '/' + ft.get_file_name_without_extension(fpath) + '_summary.xlsx')


if __name__ == '__main__':
    main()