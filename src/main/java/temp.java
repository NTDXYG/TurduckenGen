import java.util.*;
import java.util.stream.*;
import java.util.logging.*;
import java.lang.*;
import java.io.*;
import javax.servlet.http.HttpSession;
import org.springframework.jdbc.core.*;
import javax.sql.DataSource;
import com.alibaba.fastjson.*;
import javax.persistence.*;
import javax.persistence.criteria.*;
import org.springframework.ui.*;
import org.springframework.http.*;
public class temp { 

public List getDrinkers(DataSource engine, String name, String city) {
        JdbcTemplate connection = new JdbcTemplate(engine);
        String sql = "select * from Drinkers where name=? and city=?";
        List<Object> drinkData = connection.query(sql, new Object[]{name, city},
            new BeanPropertyRowMapper(List.class));
        List<Object> result = new ArrayList<Object>();
        for(Object r:drinkData){
            result.add(new HashMap(){{put("date", r);}});
        }
        return result;
    }}
