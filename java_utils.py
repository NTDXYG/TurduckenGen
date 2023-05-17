# -*- coding: utf-8 -*-
import re
import javalang
import nltk
import pandas as pd
from nltk import ngrams
from tqdm import tqdm

from CodeBLEU import bleu, weighted_ngram_match
from CodeBLEU import syntax_match, dataflow_match

from collections import Counter
from crystalbleu import corpus_bleu, SmoothingFunction

from java_compile import check_java_code


def get_executable_rate(hyps_list):
    executable_wrong_num = 0
    for i in tqdm(range(len(hyps_list))):
        try:
            flag = check_java_code(hyps_list[i])
            if flag == False:
                executable_wrong_num += 1
        except:
            executable_wrong_num += 1
    return (len(hyps_list) - executable_wrong_num)/len(hyps_list)

def find_sql(code):
    QUOTED_STRING_RE = re.compile(r"(?P<quote>['\"])(?P<string>.*?)(?P=quote)")
    matches = re.findall(QUOTED_STRING_RE, code)
    result = ""
    if len(matches) > 0:
        for i in range(len(matches)):
            if 'select' in matches[i][1] and 'from' in matches[i][1]:
                result = matches[i][1]
                break
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
                if sql_hyp == sql_gold:
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
                if sql_hyp == sql_gold:
                    em += 1
    if(count==0):
        count = 1
    # print(count)
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

def get_codebleu_score(hyp_list, ref_list):
    ref_list = [[ref] for ref in ref_list]

    # calculate ngram match (BLEU)
    tokenized_hyps = [x.split() for x in hyp_list]
    tokenized_refs = [[x.split() for x in reference] for reference in ref_list]

    MAXN = 4
    mc = 500
    sm_func = SmoothingFunction(epsilon=0.0001).method1
    df = pd.read_csv("D:\论文代码开源\FSE结果挖掘\dataset\Java\\train_raw.csv")
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
    keywords = [x.strip() for x in open('CodeBLEU/keywords/java.txt', 'r', encoding='utf-8').readlines()]

    def make_weights(reference_tokens, key_word_list):
        return {token: 1 if token in key_word_list else 0.2 \
                for token in reference_tokens}

    tokenized_refs_with_weights = [[[reference_tokens, make_weights(reference_tokens, keywords)] \
                                    for reference_tokens in reference] for reference in tokenized_refs]
    weighted_ngram_match_score = weighted_ngram_match.corpus_bleu(tokenized_refs_with_weights, tokenized_hyps)

    syntax = syntax_match.corpus_syntax_match(ref_list, hyp_list, 'java')
    dataflow = dataflow_match.corpus_dataflow_match(ref_list, hyp_list, 'java')
    codebleu_score = 0.25*ngram_match_score + 0.25*weighted_ngram_match_score + 0.25*syntax + 0.25*dataflow
    return ngram_match_score, weighted_ngram_match_score, crystalbleu_score, syntax, dataflow, codebleu_score

def get_var_replacing(code_string, repalce_string):
    code = code_string.strip()
    try:
        tokens = list(javalang.tokenizer.tokenize(code))
        tks = []
        for tk in tokens:
            tks.append({'value': tk.value, 'type': tk.__class__.__name__})
        var_dict = {}
        token_list = []
        var_index = 0
        for i in tks:
            if not repalce_string:
                if i['type'] == 'String' and re.findall(r"( FROM )|( from )", i['value'])!=[]:
                    sql_parsed = i['value']
                    token_list.append(sql_parsed)
                else:
                    if i['value'] in var_dict.keys():
                        token_list.append(var_dict[i['value']])
                    else:
                        var = "var_"+str(var_index)
                        var_dict[i['value']] = var
                        token_list.append(var)
                        var_index+=1
    except:
        token_list = []
    return token_list

def get_func_correctness(hyps_list, gold_list, repalce_string=False, need_index=False):
    ast_match_num = 0
    index = 0
    index_list = []
    for i, j in zip(hyps_list, gold_list):
        if '<unk>' not in i:
            i, j = get_var_replacing(i, repalce_string), get_var_replacing(j, repalce_string)
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
    ngram_match_score, weighted_ngram_match_score, crystalbleu_score, syntax, dataflow, codebleu = get_codebleu_score(hyp_list, ref_list)
    sql_em = get_sql_em(hyp_list, ref_list)
    executable_rate = get_executable_rate(ref_list)
    result['bleu'] = round(bleu * 100, 2)
    result['Weighted BLEU'] = round(weighted_ngram_match_score * 100, 2)
    result['Crystal BLEU'] = round(crystalbleu_score * 100, 2)
    result['Syntax Match'] = round(syntax * 100, 2)
    result['codebleu'] = codebleu
    result['Syntax Exact Matching'] = round(get_func_correctness(hyp_list, ref_list) * 100, 2)
    result['Executable'] = round(executable_rate * 100, 2)
    print(result)
    return result