import os
import random
import time
from pylint import epylint as lint

def code_staticAnaylsis(code, id=None):
    cur_time = str(time.time()).replace(".","")
    if(id is None):
        id = random.randint(1, 1000)
    with open(cur_time+str(id)+".py",'w') as f:
        f.write("# pylint: disable=E1101\n")
        f.write(code)
    (pylint_stdout, pylint_stderr) = lint.py_run(cur_time+str(id)+".py -s yes", return_std=True)
    os.remove(cur_time+str(id)+".py")
    pylint_stdout_str = pylint_stdout.read()
    if "E0" in pylint_stdout_str or "E1" in pylint_stdout_str:
        return False
    return True

if __name__ == '__main__':
    code = '''
from flask import jsonify
def book_api(conn,isbn):
	res = conn.execute("SELECT user FROM books WHERE isbn = :isbn", {"isbn": isbn}).fetchone()
	if res is None:
		return jsonify({"error": "The book is not in the database"})
	else:
		return res
    '''
    print(code_staticAnaylsis(code))