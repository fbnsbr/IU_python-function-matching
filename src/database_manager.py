import sqlalchemy as db
import pandas as pd
from src.exception_handling import FileError

class DatabaseManager:
    """
    Database Manager Class:
    Holds Methods to create a SQL DB engine, save data to the file,
    and load data from the file, as well es inspect the contents
    of the database for testing.
    """
    def __init__(self, filepath=r".\data\database.db") -> None:
        """
        Initialize DatabaseManager with optional sql filepath.
        """
        self.filepath = filepath
        try:
            self.engine = db.create_engine(f"sqlite:///{self.filepath}")
        except Exception as e:
            raise FileError("Could not access SQL database file.") from e

    def save_dataframe(self, data:pd.DataFrame, table_name:str):
        """
        Save DataFrame to sql file under given table name. Data will be overridden!
        """
        try:
            data.to_sql(table_name,
                    self.engine,
                    if_exists="replace",
                    index=False)
        except Exception as e:
            raise FileError("Could not access SQL database file.") from e

        
    def load_table(self, table_name:str) -> pd.DataFrame:
        """
        Load given table from SQL file.
        """
        try:
            return pd.read_sql_table(table_name, self.engine)
        except Exception as e:
            raise FileError("Could not access SQL database file.") from e

    def inspect_contents(self):
        """
        Inspect all Tables in the SQL file.
        """
        print(f"Contents of SQL Database at {self.filepath}")
        inspector = db.inspect(self.engine)
        tables = inspector.get_table_names()
        for table in tables:
            print(f"\n\n---- {table} ----\n")
            df = pd.read_sql_table(table, self.engine)
            print(df.head())
