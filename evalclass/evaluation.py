import os
import pandas as pd
import nltk
from nltk.tokenize import TreebankWordTokenizer
import util.bleu_score as bs
import util.rouge_score as rs
import util.nlp_util as nlp
import util.string_util as st


class Load:
    def __init__(self, path):
        self.path = path
        self.df = pd.DataFrame()
        self._corpus_list = []

    def __len__(self):
        return len(self._corpus_list)

    def __repr__(self):
        if self.df.empty:
            return 'Not loaded.'
        else:
            return 'Total Count: {}'.format(len(self.df))

    def __getitem__(self, position):
        return self._corpus_list[position]

    def read(self, path):
        extension = os.path.splitext(path)[1][1:]

        if extension == 'csv':
            self.df = pd.read_csv(path)
        elif extension == 'xlsx':
            self.df = pd.read_excel(path)

    # def write(self, path):
    #     self.df.to_excel(path, index=False)


class Evaluation:
    def __init__(self, src_lang, tgt_lang, place, method, character):
        # self.config = config
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.place = place
        self.method = method
        self.character = character
        # print(config)

    def scoring(self, df: pd.DataFrame):
        df.dropna()

        self.bleu_scoring = bs.BleuModule()
        self.rouge_scoring = rs.RougeModule()

        df['tgt'] = df.apply(lambda row: self.__clean(row.tgt), axis=1)
        ref = self.__tokenize(df['tgt'], self.tgt_lang)

        col_list = list(df.columns[self.place:])
        print('Start Evaluation ---')
        for col in col_list:
            df[col] = [self.__clean(t) for t in df[col]]
            hyp = self.__tokenize(df[col], self.tgt_lang)

            eval_df = pd.DataFrame({'ref':ref,
                                    'hyp':hyp})

            if 'a' in self.method:
                print('ALL Metrics Evaluation : BLEU, BLEU-1, ROUGE ---')
                df[f'{col}_bleu'] = eval_df.apply(lambda row: self.__machine_evaluation(row.ref, row.hyp, method='bleu'), axis=1)
                print(f'{col} BLEU Scoring Finish')
                df[f'{col}_bleu1'] = eval_df.apply(lambda row: self.__machine_evaluation(row.ref, row.hyp, method='bleu1'), axis=1)
                print(f'{col} BLEU-UNI Scoring Finish')
                df[f'{col}_rouge'] = eval_df.apply(lambda row: self.__machine_evaluation(row.hyp, row.ref, method='rouge'), axis=1)
                print(f'{col} ROUGE Scoring Finish')
            elif 'r' in self.method:
                print('Metrics Evaluation : ROUGE ---')
                df[f'{col}_rouge'] = eval_df.apply(lambda row: self.__machine_evaluation(row.hyp, row.ref, method='rouge'), axis=1)
                print(f'{col} ROUGE Scoring Finish')
            elif 'u' in self.method:
                print('Metrics Evaluation : BLEU-1 ---')
                df[f'{col}_bleu1'] = eval_df.apply(lambda row: self.__machine_evaluation(row.ref, row.hyp, method='bleu1'), axis=1)
                print(f'{col} BLEU-UNI Scoring Finish')
            elif 'b' in self.method:
                print('Metrics Evaluation : BLEU ---')
                df[f'{col}_bleu'] = eval_df.apply(lambda row: self.__machine_evaluation(row.ref, row.hyp, method='bleu'), axis=1)
                print(f'{col} BLEU Scoring Finish')

        return df

    def __machine_evaluation(self, ref, hyp, method: str):
        try:
            if method == 'rouge':
                eval_score = self.rouge_scoring.calculate(hyp, ref)
                result = self.__norm_score(eval_score)
            elif method == 'bleu1':
                eval_score = self.bleu_scoring.calculate(ref, hyp, weight=(0, 1, 0, 0))
                result = self.__norm_score(eval_score)
            elif method == 'bleu':
                eval_score = self.bleu_scoring.calculate(ref, hyp)
                result = self.__norm_score(eval_score)
            else:
                result = []
                print('Unknown method ---')
        except Exception as e:
            result = []
            print('Error ---', e)
        return result

    def __clean(self, text):
        text = str(text).replace('\n', '')
        text = str(text).replace('\t', '')
        text = text.strip()
        text = st.unescaped(text)
        return text

    def __tokenize(self, l, tgt_lang: str):
        if tgt_lang == 'ko':
            if self.character == 'T':
                token_list = nlp.morphs_cjk(l)
            elif self.character == 'F':
                token_list = nlp.morphs_mecab(l)
        elif tgt_lang == 'de':
            token_list = nlp.german_nltk_tokenizer(l)
        elif tgt_lang == 'ja':
            token_list = nlp.morphs_cjk(l)
        elif tgt_lang == 'zh':
            token_list = nlp.morphs_cjk(l)
        elif tgt_lang == 'en':
            token_list = nlp.morph_nltk(str(l))
        elif tgt_lang == 'es' or tgt_lang == 'fr':
            token_list = [nltk.tokenize.word_tokenize(l[i]) for i in range(len(l))]
        else:
            token_list = []
            print('No language code.')
        return token_list

    def __norm_score(self, sc):
        try:
            norm = round(sc * 100, 2)
        except TypeError:
            norm = 'None'
        return norm

    def make_summary(self, df):
        summary = pd.DataFrame(df.mean().round(4), columns=['score'])
        col = [s.split('_')[1] for s in summary.index]

        bleu = [summary['score'][i] for i in range(len(summary)) if 'bleu' in col[i] and '1' not in col[i]]
        bleu1 = [summary['score'][i] for i in range(len(summary)) if 'bleu1' in col[i]]
        rouge = [summary['score'][i] for i in range(len(summary)) if 'rouge' in col[i]]

        result_summary = pd.DataFrame({'BLEU': bleu,
                                       'BLEU1': bleu1,
                                       'ROUGE': rouge})
        return result_summary


class Simple_Evaluation:
    def __init__(self, src_lang, tgt_lang, col_names, method, pos):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.col_names = col_names
        self.method = method
        self.pos = pos

    def scoring(self, df: pd.DataFrame):
        df.dropna()

        self.bleu_scoring = bs.BleuModule()
        self.rouge_scoring = rs.RougeModule()

        df['tgt'] = df.apply(lambda row: self.__clean(row.tgt), axis=1)
        ref = self.__tokenize(df['tgt'], self.tgt_lang)

        col_list = [self.col_names]
        col_list = [c for c in col_list[0].split(',')]
        print(col_list)
        print('Start Evaluation ---')

        for col in col_list:
            df[col] = [self.__clean(t) for t in df[col]]
            hyp = self.__tokenize(df[col], self.tgt_lang)

            eval_df = pd.DataFrame({'ref':ref,
                                    'hyp':hyp})

            if 'a' in self.method:
                print('ALL Metrics Evaluation : BLEU, BLEU-1, ROUGE ---')
                df[f'{col}_bleu'] = eval_df.apply(lambda row: self.__machine_evaluation(row.ref, row.hyp, method='bleu'), axis=1)
                print(f'{col} BLEU Scoring Finish')
                df[f'{col}_bleu1'] = eval_df.apply(lambda row: self.__machine_evaluation(row.ref, row.hyp, method='bleu1'), axis=1)
                print(f'{col} BLEU-UNI Scoring Finish')
                df[f'{col}_rouge'] = eval_df.apply(lambda row: self.__machine_evaluation(row.hyp, row.ref, method='rouge'), axis=1)
                print(f'{col} ROUGE Scoring Finish')
            elif 'r' in self.method:
                print('Metrics Evaluation : ROUGE ---')
                df[f'{col}_rouge'] = eval_df.apply(lambda row: self.__machine_evaluation(row.hyp, row.ref, method='rouge'), axis=1)
                print(f'{col} ROUGE Scoring Finish')
            elif 'u' in self.method:
                print('Metrics Evaluation : BLEU-1 ---')
                df[f'{col}_bleu1'] = eval_df.apply(lambda row: self.__machine_evaluation(row.ref, row.hyp, method='bleu1'), axis=1)
                print(f'{col} BLEU-UNI Scoring Finish')
            elif 'b' in self.method:
                print('Metrics Evaluation : BLEU ---')
                df[f'{col}_bleu'] = eval_df.apply(lambda row: self.__machine_evaluation(row.ref, row.hyp, method='bleu'), axis=1)
                print(f'{col} BLEU Scoring Finish')

        return df

    def __machine_evaluation(self, ref, hyp, method: str):
        try:
            if method == 'rouge':
                eval_score = self.rouge_scoring.calculate(hyp, ref)
                result = self.__norm_score(eval_score)
            elif method == 'bleu1':
                eval_score = self.bleu_scoring.calculate(ref, hyp, weight=(0, 1, 0, 0))
                result = self.__norm_score(eval_score)
            elif method == 'bleu':
                eval_score = self.bleu_scoring.calculate(ref, hyp)
                result = self.__norm_score(eval_score)
            else:
                result = []
                print('Unknown method ---')
        except Exception as e:
            result = []
            print('Error ---', e)
        return result

    def __clean(self, text):
        text = str(text).replace('\n', '')
        text = str(text).replace('\t', '')
        text = text.strip()
        text = st.unescaped(text)
        return text

    def __tokenize(self, l, tgt_lang: str):
        if tgt_lang == 'ko':
            if self.pos == 'y':
                token_list = nlp.morphs_okt(l)
            elif self.pos == 'n':
                token_list = nlp.morphs_cjk(l)
        elif tgt_lang == 'de':
            token_list = nlp.german_nltk_tokenizer(l)
        elif tgt_lang == 'ja':
            token_list = nlp.morphs_cjk(l)
        elif tgt_lang == 'zh':
            token_list = nlp.morphs_cjk(l)
        elif tgt_lang == 'en':
            obj = TreebankWordTokenizer()
            token_list = [obj.tokenize(text) for text in l]
        elif tgt_lang == 'es' or tgt_lang == 'fr':
            token_list = [nltk.tokenize.word_tokenize(l[i]) for i in range(len(l))]
        elif tgt_lang == 'vi':
            token_list = [text.split() for text in l]
        else:
            token_list = []
            print('No language code.')
        return token_list

    def __norm_score(self, sc):
        try:
            norm = round(sc * 100, 2)
        except TypeError:
            norm = 'None'
        return norm

    def make_summary(self, df):
        summary = pd.DataFrame(df.mean().round(4), columns=['score'])
        col = [s.split('_')[-1] for s in summary.index]

        bleu = [summary['score'][i] for i in range(len(summary)) if 'bleu' in col[i] and '1' not in col[i]]
        bleu1 = [summary['score'][i] for i in range(len(summary)) if 'bleu1' in col[i]]
        rouge = [summary['score'][i] for i in range(len(summary)) if 'rouge' in col[i]]

        result_summary = pd.DataFrame({'BLEU': bleu,
                                       'BLEU1': bleu1,
                                       'ROUGE': rouge})
        return result_summary