from google.cloud.sql.connector import Connector
import sqlalchemy
import pandas as pd

# initialize parameters
INSTANCE_CONNECTION_NAME = "testanalyticsplatform:europe-west3:data-monkey-test-2" # i.e demo-project:us-central1:demo-instance
print(f"Your instance connection name is: {INSTANCE_CONNECTION_NAME}")
DB_USER = "postgres"
DB_PASS = "Datamonkeytest123"
DB_NAME = "postgres"

import sqlalchemy
import pandas as pd

class DB_Connector:
    def getconn(self):
        connector = Connector()
        conn = connector.connect(
            INSTANCE_CONNECTION_NAME,
            "pg8000",
            user=DB_USER,
            password=DB_PASS,
            db=DB_NAME
        )
        return conn

    def enlist_tables(self, conn):
        engine = sqlalchemy.create_engine(
            "postgresql+pg8000://",
            creator=lambda: conn,  # Pass the connection object directly
        )
        return engine

if __name__ == "__main__":
    db_connector = DB_Connector()
    conn = db_connector.getconn()
    engine = db_connector.enlist_tables(conn)

    # Now use the engine object to execute queries
    employees = pd.read_sql_query("SELECT name, title, salary from employees", engine)
    print(employees.head(30))