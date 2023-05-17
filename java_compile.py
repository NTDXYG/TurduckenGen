import subprocess

def check_java_code(code):
    program = 'import java.util.*;\n' \
              'import java.util.stream.*;\n' \
              'import java.util.logging.*;\n' \
              'import java.lang.*;\n' \
              'import java.io.*;\n' \
              'import javax.servlet.http.HttpSession;\n' \
              'import org.springframework.jdbc.core.*;\n' \
              'import javax.sql.DataSource;\n' \
              'import com.alibaba.fastjson.*;\n' \
              'import javax.persistence.*;\n' \
              'import javax.persistence.criteria.*;\n' \
              'import org.springframework.ui.*;\n' \
              'import org.springframework.http.*;'
    program += '\n'
    program += 'public class temp { \n\n'
    program += code
    program += '}\n'
    with open("src/main/java/temp.java", 'w', encoding='utf8') as f:
        f.write(program)
    command = ["mvn", "compile"]
    try:
        result = subprocess.getoutput(command)
        if 'BUILD SUCCESS' in result:
            return True
        else:
            return False
    except:
        return False

if __name__ == '__main__':
    code = '''
    public Object apiBookInfo ( JdbcTemplate conn , String isbn ) { 
        List books = conn . query ( " select user from books where isbn = ? " , new Object [ ] { isbn } , 
        new BeanPropertyRowMapper ( List . class ) ) ; 
        Object book = books . get ( 0 ) ; 
        if ( books . isEmpty ( ) ) { 
            JSONObject json = new JSONObject ( ) ; 
            json . put ( " error " , " The book is not in the database " ) ; 
            return json ; 
        } else { 
            return JSONObject . parseObject ( book . toString ( ) ) ;
        } 
    }
    '''
    check_java_code(code)