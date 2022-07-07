from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from joblib import Parallel, delayed
import multiprocessing as mp

# Prerequisite: pip3 install konlpy, pip3 install nltk
# The score range is 0 from 1: perfect match result is 1, whereas a perfect mismatch is 0.


class BleuModule:
    def __init__(self, weight=(0.25,0.25,0.25,0.25)):
        self.weight = weight

    def calculate_parallel(self, zipped, num_cores=mp.cpu_count(), verbose_status=0):
        if zipped is None:
            return None
        else:
            return Parallel(n_jobs=num_cores, verbose=verbose_status)(delayed(self.calculate_zipped)(i) for i in zipped)

    def calculate_zipped(self, zipped):
        # token size should be greater than 1
        if len(zipped[0]) <= 1 or len(zipped[1]) <= 1:
            return 0.0
        else:
            wrapped_res = [zipped[0]]
            smoothie = SmoothingFunction().method3
            return sentence_bleu(wrapped_res, zipped[1], weights=self.weight, smoothing_function=smoothie)

    def calculate(self, reference, hypothesis, weight=(0.25,0.25,0.25,0.25)):
        wrapped_res = [reference]
        smoothie = SmoothingFunction().method3
        try:
            score = sentence_bleu(wrapped_res, hypothesis, weights= weight, smoothing_function=smoothie)
        except Exception as e:
            print('Error ---', e)
            score = 'None'
        return score