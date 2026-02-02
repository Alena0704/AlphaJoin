import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;

public class JDBCUtil {
    private static final String DRIVER = "org.postgresql.Driver";
    // Подключаемся к вашему инстансу Postgres на порту 5496, БД postgres, роль boo (как в psql).
    private static final String URL = "jdbc:postgresql://localhost:5496/postgres";
    private static final String NAME = "boo";
    private static final String PASSWORD = "";
    private static Connection conn = null;


    static{
        try {
            Class.forName(DRIVER);
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
    }


    public static Connection getConnection() {
        try {
            return DriverManager.getConnection(URL, NAME, PASSWORD);
        } catch (SQLException e) {
            e.printStackTrace();
        }
        return null;
    }

}
