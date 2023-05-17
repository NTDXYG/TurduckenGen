# -*- coding: utf-8 -*-
import random
import time
import os
import re

import nltk
import pandas as pd
import parso
from nltk import ngrams
from parso.python import tokenize
from tqdm import tqdm

from python_compile import code_staticAnaylsis

version_info = parso.utils.parse_version_string("3.8")
# from nltk.translate import bleu_score
# from nltk.translate.bleu_score import corpus_bleu
from pylint import epylint as lint
from CodeBLEU import bleu, weighted_ngram_match, syntax_match, dataflow_match

from collections import Counter
# Import CrystalBLEU
from crystalbleu import corpus_bleu, SmoothingFunction

def find_sql(code):
    QUOTED_STRING_RE = re.compile(r"(?P<quote>['\"])(?P<string>.*?)(?P=quote)")
    matches = re.findall(QUOTED_STRING_RE, code)
    result = ""
    if len(matches) > 0:
        for i in range(len(matches)):
            if 'select' in matches[i][1] and 'from' in matches[i][1]:
                result = matches[i][1]
                break
    result = result.replace('=', ' = ')
    result = result.replace('  ', ' ')
    # result = result.lower()
    return result

def get_sql_em(hyps_list, gold_list, analysis=False):
    data_list = []
    count = 0
    em = 0
    if analysis == True:
        for i in range(len(gold_list)):
            sql_gold = find_sql(gold_list[i])
            if len(sql_gold)>0:
                count += 1
                sql_hyp = find_sql(hyps_list[i])
                if sql_hyp.strip() == sql_gold.strip():
                    em += 1
                else:
                    data_list.append([sql_hyp, sql_gold])
        df = pd.DataFrame(data_list, columns=['hyp', 'ref'])
        df.to_csv("sql.csv")
    else:
        for i in range(len(gold_list)):
            sql_gold = find_sql(gold_list[i])
            if len(sql_gold)>0:
                count += 1
                sql_hyp = find_sql(hyps_list[i])
                if sql_hyp.strip() == sql_gold.strip():
                    em += 1
    return em/count

def get_em(hyps_list, gold_list):
    count = len(hyps_list)
    em = 0
    for i in range(len(gold_list)):
        if hyps_list[i].strip() == gold_list[i].strip():
            em += 1
    return em/count

# evaluation metrics
def get_bleu4_score(hyps_list, gold_list, tokenizer=None):
    if tokenizer==None:
        b_score = corpus_bleu(
            [[ref.split()] for ref in gold_list],
            [pred.split() for pred in hyps_list],
            smoothing_function = nltk.translate.bleu_score.SmoothingFunction(epsilon=1e-12).method1,
            weights=(0.25, 0.25, 0.25, 0.25))
    else:
        b_score = corpus_bleu(
            [[tokenizer.tokenize(ref)] for ref in gold_list],
            [tokenizer.tokenize(pred) for pred in hyps_list],
            smoothing_function = nltk.translate.bleu_score.SmoothingFunction(epsilon=1e-12).method1,
            weights=(0.25, 0.25, 0.25, 0.25))
    return b_score

def get_executable_rate(hyps_list):
    executable_wrong_num = 0
    for i in tqdm(range(len(hyps_list))):
        if '<unk>' not in hyps_list[i]:
            if code_staticAnaylsis(hyps_list[i].replace("\t","    "), i) == False:
                executable_wrong_num+=1
        else:
            executable_wrong_num+=1
    return (len(hyps_list) - executable_wrong_num)/len(hyps_list)

def get_codebleu_score(hyp_list, ref_list):
    ref_list = [[ref] for ref in ref_list]

    # calculate ngram match (BLEU)
    tokenized_hyps = [x.split() for x in hyp_list]
    tokenized_refs = [[x.split() for x in reference] for reference in ref_list]

    MAXN = 4
    mc = 500
    sm_func = SmoothingFunction(epsilon=0.0001).method1
    df = pd.read_csv("..\dataset\Python\\train.csv")
    data = df['code'].tolist()
    all_ngrams = []
    total_tokens = 0
    for j in data:
        tokenized = j.split()
        total_tokens += len(tokenized)
        for j in range(1, MAXN+1):
            n_grams = list(ngrams(tokenized, j))
            all_ngrams.extend(n_grams)
    freq = Counter(all_ngrams)
    comm_ngrams = dict(freq.most_common(mc))
    most_common_dict = comm_ngrams
    crystalbleu_score = corpus_bleu(tokenized_refs, tokenized_hyps, smoothing_function=sm_func, ignoring=most_common_dict)

    ngram_match_score = bleu.corpus_bleu(tokenized_refs, tokenized_hyps)

    # calculate weighted ngram match
    keywords = [x.strip() for x in open('/CodeBLEU\keywords\python.txt', 'r', encoding='utf-8').readlines()]

    def make_weights(reference_tokens, key_word_list):
        return {token: 1 if token in key_word_list else 0.2 \
                for token in reference_tokens}

    tokenized_refs_with_weights = [[[reference_tokens, make_weights(reference_tokens, keywords)] \
                                    for reference_tokens in reference] for reference in tokenized_refs]
    weighted_ngram_match_score = weighted_ngram_match.corpus_bleu(tokenized_refs_with_weights, tokenized_hyps)

    syntax = syntax_match.corpus_syntax_match(ref_list, hyp_list, 'python')
    dataflow = dataflow_match.corpus_dataflow_match(ref_list, hyp_list, 'python')
    codebleu_score = 0.25*ngram_match_score + 0.25*weighted_ngram_match_score + 0.25*syntax + 0.25*dataflow
    return ngram_match_score, weighted_ngram_match_score, crystalbleu_score, syntax, dataflow, codebleu_score

def get_var_replacing(code_string, repalce_string):
    version_info = parso.utils.parse_version_string("3.8")
    var_dict = {}
    token_list = []
    var_index = 0
    for i in tokenize.tokenize(code_string, version_info):
        if not repalce_string:
            # print(i)
            if i.type == tokenize.STRING and re.findall(r"( FROM )|( from )", i.string)!=[]:
                sql_parsed = i.string
                token_list.append(sql_parsed)
            else:
                if i.string in var_dict.keys():
                    token_list.append(var_dict[i.string])
                else:
                    var = "var_"+str(var_index)
                    var_dict[i.string] = var
                    token_list.append(var)
                    var_index+=1

            # else:
            #     token_list.append(i.string)
        else:
            if i.string in var_dict.keys():
                token_list.append(var_dict[i.string])
            else:
                var = "var_" + str(var_index)
                var_dict[i.string] = var
                token_list.append(var)
                var_index += 1
            # if i.type == tokenize.NAME or (i.type == tokenize.STRING and re.findall(r"( FROM )|( from )", i.string)!=[]):
            #     if i.string in var_dict.keys():
            #         token_list.append(var_dict[i.string])
            #     else:
            #         var = "var_"+str(var_index)
            #         var_dict[i.string] = var
            #         token_list.append(var)
            #         var_index+=1
            # else:
            #     token_list.append(i.string)
    return token_list


def get_func_correctness(hyps_list, gold_list, repalce_string=False, need_index=False):
    ast_match_num = 0
    index = 0
    index_list = []
    for i, j in zip(hyps_list, gold_list):
        if '<unk>' not in i:
            i, j = get_var_replacing(i, repalce_string), get_var_replacing(j, repalce_string)
            # print(i)
            if i == j:
                ast_match_num+=1
                index_list.append(index)
        index+=1
    # print("Number of AST matching", ast_match_num)
    # print("Accuration of AST matching", ast_match_num/len(hyps_list))
    if need_index==True:
        return ast_match_num/len(hyps_list), " ".join([str(k) for k in index_list])
    else:
        return ast_match_num/len(hyps_list)

def compute_all_metrics(hyp_list, ref_list):
    result = {}
    bleu = get_bleu4_score(hyp_list, ref_list)
    ngram_match_score, weighted_ngram_match_score, crystalbleu_score, syntax, dataflow, codebleu_score = get_codebleu_score(hyp_list, ref_list)
    # sql_em = get_sql_em(hyp_list, ref_list, analysis=False)
    em = get_em(hyp_list, ref_list)
    result['bleu'] = bleu
    result['Weighted BLEU'] = weighted_ngram_match_score
    result['Crystal BLEU'] = crystalbleu_score
    result['Syntax Match'] = syntax
    result['Dataflow Match'] = dataflow
    result['Syntax Exact Matching'] = get_func_correctness(hyp_list, ref_list)
    result['CodeBLEU'] = codebleu_score
    print(result)
    executable_rate = get_executable_rate(hyp_list)
    result['Executable'] = executable_rate
    print(executable_rate)
