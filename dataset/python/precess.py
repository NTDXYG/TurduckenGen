import re
import pandas as pd
import parso
from parso.python import tokenize
from asts.ast_parser import parse_ast, get_xsbt

QUOTED_STRING_RE = re.compile(r"(?P<quote>['\"])(?P<string>.*?)(?P=quote)")
D_STRING_RE = re.compile(r"(?P<quote>[$])(?P<string>.*?)(?P=quote)")
version_info = parso.utils.parse_version_string("3.8")

def clean_code(code):
    code = code.strip()
    code = code.replace('\t', '!TAB!')
    code_list = code.split('\n')
    result_list = []
    for code in code_list:
        result = []
        version_info = parso.utils.parse_version_string("3.8")
        for i in tokenize.tokenize(code, version_info):
            result.append(i.string)
        result_list.append(' '.join(result).strip())
    result = '\n'.join(result_list)
    result = result.replace('SELECT', 'select')
    result = result.replace('FROM', 'from')
    result = result.replace('WHERE', 'where')
    result = result.replace('AND', 'and')
    result = result.replace('\'', '"')
    result = result.replace('"', ' " ')
    result = result.replace('! TAB !', '\t')
    result = result.replace('< :', ' < : ')
    result = result.replace('<:', ' < : ')
    result = result.replace('> :', ' > : ')
    result = result.replace('>:', ' > : ')
    result = result.replace('= :', ' = : ')
    result = result.replace('=:', ' = : ')
    result = result.replace('  ', ' ')
    result = result.replace('  ', ' ')
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

df = pd.read_csv('Lyra-python-'+type+'.csv')
code_list = df['code'].tolist()
nl_list = df['comm_en'].tolist()

data_list = []

for i in range(len(code_list)):
    nl = clean_token(nl_list[i])
    code = clean_code(code_list[i])

    xsbt = get_xsbt(code, parse_ast(code, 'python'), 'python')
    xsbt = ' '.join(xsbt)
    xsbt = xsbt.replace('<str> " " </str>', 'STR')

    data_list.append(['Generate origin code: '+nl, code,
                      'Generate syntax code: '+nl, xsbt])


df = pd.DataFrame(data_list, columns=['nl', 'code', 'syntax_nl', 'syntax_code'])
df.to_csv(type + ".csv", index=False)
