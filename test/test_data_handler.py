import unittest
import pandas as pd
from pathlib import Path
from src.data_handler import (DataHandler,
    TrainDataHandler, IdealDataHandler, TestDataHandler)

from src.exception_handling import (DataValidationError,
    DataAlignmentError, FileError)

class UnitTestDataHandler(unittest.TestCase):
    """
    TestCase-derived Class to test DataHandler Class
    """
    def setUp(self):
        """
        Set up the common requirements for each independent test
        Create csv file, create DataHandler and load with data
        """
        self.test_file = "temp.csv"
        self.test_data = pd.DataFrame({
                        "x":[1,2],
                        "y":[3,4]  })
        self.test_data.to_csv(self.test_file, index=False)
        self.handler = DataHandler(self.test_file, "x","y_{i}")
        self.handler.load()

    def tearDown(self):
        """
        Delete the csv file from SetUp method
        """
        Path(self.test_file).unlink(missing_ok=True)
    
    def test_load_notempty(self):
        # confirm file loaded
        self.assertFalse(self.handler.data.empty)
        
    def test_load_missing_file(self):
        # try to load missing file
        self.handler2 = DataHandler("invalid.csv")
        with self.assertRaises(FileError):
            self.handler2.load()

    def test_validate_empty(self):
        # empty dataframe
        self.handler.data = pd.DataFrame({})
        with self.assertRaises(DataValidationError):
            self.handler.validate()

    def test_validate_xaxis(self):
        # invalid x-axis
        self.handler.Xaxis=""
        with self.assertRaises(DataValidationError):
            self.handler.validate() 

    def test_validate_na(self):
        # one field NA
        self.handler.data.iloc[0,0] = pd.NA
        with self.assertRaises(DataValidationError):
            self.handler.validate()          
        
    def test_rename_columns(self):
        # confirm y columns renamed and names saved
        self.handler.rename_columns()
        self.assertEqual(self.handler.function_names,["y_1"])

    def test_align_inconsistent_xaxis(self):
        # try to merge with inconsistent x-axis
        handler2 = DataHandler(self.test_file, "not_x","y_{i}")
        handler2.load()
        with self.assertRaises(DataAlignmentError):
            self.handler.align_with(handler2)
    
    def test_align_not_one_on_one(self):
        # try to merge with duplicate x-values, not 1:1
        handler2 = DataHandler(self.test_file, "x","y_{i}")
        handler2.load()
        #create duplicate x-value in row 1 and row 2
        handler2.data.loc[0, handler2.Xaxis] = handler2.data.loc[1, handler2.Xaxis]
        with self.assertRaises(DataAlignmentError):
            self.handler.align_with(handler2)

    def test_align_losing_data(self):
        # try to merge with one losing one x value on the left
        handler2 = DataHandler(self.test_file, "x","y_{i}")
        handler2.load()
        self.handler.data.loc[0,self.handler.Xaxis] = 999
        with self.assertRaises(DataAlignmentError):
            self.handler.align_with(handler2)

class UnitTestTrainDataHandler(UnitTestDataHandler):
    """
    Subclass to test TrainDataHandler Class
    """
    def setUp(self):
        """
        Set up the common requirements for each independent test
        Create csv file, create DataHandler and load with data
        """
        # as TestData, create a 5x400 dataframe filled with numbers
        self.test_file = "temp.csv"
        series = pd.Series(range(0,400))
        self.test_data = pd.DataFrame({"x": series})
        for i in range(1,5):
            self.test_data[f"y{i}"] = series
        self.test_data.to_csv(self.test_file, index=False)
        
        self.handler = TrainDataHandler(self.test_file, "x","y_{i}")
        self.handler.load()

    def tearDown(self):
        super().tearDown()

    def test_validate_columns(self):
        # test wrong number of columns
        self.handler.data["new_column"] = self.handler.data[self.handler.Xaxis]
        with self.assertRaises(DataValidationError):
            self.handler.validate()

    def test_rename_columns(self):
        # confirm y columns renamed and names saved
        self.handler.rename_columns()
        self.assertEqual(self.handler.function_names,
                         ["y_1","y_2","y_3","y_4"])
    
class UnitTestIdealDataHandler(UnitTestDataHandler):
    """
    Subclass to test IdealDataHandler Class
    """
    def setUp(self):
        # as IdealData, create a 51x400 dataframe filled with numbers
        self.test_file = "temp.csv"
        series = pd.Series(range(0,400))
        self.test_data = pd.DataFrame({"x": series})
        for i in range(1,51):
            self.test_data[f"y{i}"] = series
        self.test_data.to_csv(self.test_file, index=False)
        self.handler = IdealDataHandler(self.test_file, "x","y_{i}")
        self.handler.load()

    def tearDown(self):
        super().tearDown()
        
    def test_rename_columns(self):
        # confirm y columns renamed and names saved
        self.handler.rename_columns()
        self.assertEqual(len(self.handler.function_names),
                        50)


class UnitTestTestDataHandler(UnitTestDataHandler):
    """
    Subclass to test TestDataHandler Class
    """
    def setUp(self):
        # as TestData, create a 2x100 dataframe with x,y-columnds and filled with numbers
        # since TestData may have duplicate x-values, that is simulated by duplication of a 2x50 dataframe
        self.test_file = "temp.csv"
        series = pd.Series(range(0,50))
        self.test_data = pd.DataFrame({"x": series, "y":series})
        self.test_data = pd.concat([self.test_data, self.test_data])
        self.test_data.to_csv(self.test_file, index=False)
        self.handler = TestDataHandler(self.test_file)
        self.handler.load()

    def tearDown(self):
        super().tearDown()
        
    def test_validate_duplicate_x(self):
        # x-y test data may have duplicate x values
        self.handler.validate()

    def test_rename_columns(self):
        # confirm x,y columns not renamed
        self.handler.rename_columns()
        self.assertEqual(self.handler.data.columns.to_list(),
                         ["x","y"])


if __name__ == "__main__":
    unittest.main()