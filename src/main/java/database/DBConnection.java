package database;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.Statement;
import java.sql.SQLException;



public class DBConnection {
	private Connection con; //커넥션 객체
	private Statement st; //statementc 특정 db sql 문장을 실행하는 방법
	private ResultSet rs; //결과를 받아오는 객체
   
	public DBConnection() {
		try
		{
	         Class.forName("com.mysql.jdbc.Driver");
	         con = DriverManager.getConnection("jdbc:mysql://localhost:3306/mydatabase","root","1234");
	         st =con.createStatement();
	    }
	    catch(Exception e)
	    {
	    	System.out.println("데이터베이스 연결 오류"+ e.getMessage());
	    }
	}


}
