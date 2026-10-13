import unittest
import pandas as pd
from src.database_manager import DatabaseManager
from src.exception_handling import (
    DataValidationError, DataAlignmentError, FileError)

class UnitTestDatabaseManager(unittest.TestCase):
    """
    TestCase-derived Class to test DatabaseManager Class
    """
    def test_invalidfile(self):
        #try to write invalid file path
        with self.assertRaises(FileError):
            dbmanager = DatabaseManager(filepath="//")
            dbmanager.save_dataframe (
                pd.DataFrame({
                    "column":[1,2,3]
                }), "table")