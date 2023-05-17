import re
import pandas as pd
import javalang
from asts.ast_parser import parse_ast, get_xsbt

QUOTED_STRING_RE = re.compile(r"(?P<quote>['\"])(?P<string>.*?)(?P=quote)")
D_STRING_RE = re.compile(r"(?P<quote>[$])(?P<string>.*?)(?P=quote)")

def clean_code(code):
    code = code.strip()
    tokens = list(javalang.tokenizer.tokenize(code))
    tks = []
    for tk in tokens:
        tks.append(tk.value)
    result = " ".join(tks)
    result = result.replace('SELECT', 'select')
    result = result.replace('FROM', 'from')
    result = result.replace('WHERE', 'where')
    result = result.replace('AND', 'and')
    result = result.replace('\'', '"')
    result = result.replace('"', ' " ')
    result = result.replace('<', ' < ')
    result = result.replace('>', ' > ')
    result = result.replace('=', ' = ')
    result = result.replace('  ', ' ')
    result = result.replace('  ', ' ')
    result = result.replace('= =', '==').strip()
    result = result.replace('! =', '!=').strip()
    result = result.replace('< =', '<=').strip()
    result = result.replace('> =', '>=').strip()
    result = result.replace('+ =', '+=').strip()
    return result

def clean_token(token):
    token = token.replace(")", " ) ")
    token = token.replace("(", " ( ")
    token = token.replace("$)", "$ )")
    token = token.replace("  ", " ")
    token = token.replace('\n', ' ').strip()
    punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^`{|}~"""
    for t in token:
        if t in punctuation:
            token = token.replace(t, '')
        else:
            break
    return token.strip()

type = 'train'

df = pd.read_csv('Pisces-java-'+type+'.csv')
code_list = df['Code'].tolist()
nl_list = df['NL'].tolist()
id_list = df['id'].tolist()

data_list = []

for i in range(len(code_list)):

    nl = clean_token(nl_list[i])
    code = clean_code(code_list[i])

    xsbt = get_xsbt(code, parse_ast(code, 'java'), 'java')
    xsbt = ' '.join(xsbt)
    xsbt = xsbt.replace('<str> " " </str>', 'STR')

    data_list.append(['Generate origin code: '+nl, code,
                      'Generate syntax code: '+nl, xsbt])

df = pd.DataFrame(data_list, columns=['nl', 'code', 'syntax_nl', 'syntax_code'])
df.to_csv(type + ".csv", index=False)
