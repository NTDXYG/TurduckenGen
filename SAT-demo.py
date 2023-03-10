from asts.ast_parser import parse_ast, get_xsbt

code = 'public String hello ( JdbcTemplate conn , String id ) { List result = conn . query ( " select * from noodles where id = ? " , new Object [ ] { id } , new BeanPropertyRowMapper ( List . class ) ) ; String return_value = " " ; for ( Object row : result ) { return_value += JSONObject . parseObject ( row . toString ( ) ) . getString ( " stuff " ); } return return_value ; }'

xsbt = get_xsbt(code, parse_ast(code, 'java'), 'java')
xsbt = ' '.join(xsbt)
xsbt = xsbt.replace('<str> " " </str>', 'STR')

print(xsbt)