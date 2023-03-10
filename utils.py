# -*- coding: utf-8 -*-
import random
import time
import os
import re
import torch
import random
import numpy as np
from nltk.translate import bleu_score
from nltk.translate.bleu_score import corpus_bleu

# evaluation metrics
def get_bleu4_score(hyps_list, gold_list):
    b_score = corpus_bleu(
        [[ref.split()] for ref in gold_list],
        [pred.split() for pred in hyps_list],
        smoothing_function = bleu_score.SmoothingFunction(epsilon=1e-12).method1, 
        weights=(0.25, 0.25, 0.25, 0.25))
    return b_score

def set_seed(seed=1234):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True