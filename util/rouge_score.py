from rouge import Rouge


class RougeModule:
    '''
    prerequisite: pip install rouge (https://pypi.org/project/rouge/)
    Rouge metric
    '''

    def __init__(self, param_type='rouge-1', score_type='f'):
        '''
        :param param_type: rouge-1, rouge-2, rouge-l
        :param score_type: f (f1 score), p (stands for precision), r (stands for recall)
        '''
        self.param_type = param_type
        self.score_type = score_type
        self.rouge = Rouge()

    def calculate(self, ref, hyp):
        '''
        :param ref: tokenized list
        :param hyp: tokenized list
        :return: the score
        '''
        ref = ' '.join(ref)
        hyp = ' '.join(hyp)
        try:
            scores = self.rouge.get_scores(ref, hyp)
            result = scores[0][self.param_type][self.score_type]
        except Exception as e:
            print('Error ---', e)
            result = 0
        return result

    def calculate_zipped(self, zipped):
        '''
        calculate rouge metrics
        :param zipped: tokenized zip list of TUs
        :return: scores
        '''
        if len(zipped[0]) <= 1 or len(zipped[1]) <= 1:
            return None
        return [self.calculate(r, h) for r, h in zipped]

