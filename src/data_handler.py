import pandas as pd
from src.exception_handling import (
    DataValidationError, DataAlignmentError, FileError)

class DataHandler:
    """
    DataHandler Class holds methods to load from csv, 
    validate, rename and align (merge) data.
    """
    def __init__(self, filepath:str, xaxis:str="x", designator:str="Y{i}"):
        """
        Initialize the DataHandler Class with a csv filepath, name of x-axis column, 
        and chosen designator for the function names.
        """
        self.filepath = filepath
        self.Xaxis = xaxis
        self.designator = designator
        self.data = pd.DataFrame()
        self.function_names = []

    def load(self):
        """
        Load csv file into the objects data
        """
        try:
            self.data = pd.read_csv(self.filepath)
        except FileNotFoundError as e:
            raise FileError from e
        except PermissionError as e:
            raise FileError from e
        
    def validate(self):
        """
        Validate data: Not empty, has exactly one x-column, 
        has no missing values, has only numeric values.
        """
        if self.data.empty:
            raise DataValidationError("Dataset is empty.")
        if self.Xaxis not in self.data.columns:
            raise DataValidationError("x column missing.")
        if (self.data.columns == self.Xaxis).sum() > 1:
            raise DataValidationError("x column is duplicate.")
        if self.data.isna().any().any():
            raise DataValidationError("Missing values found.")
        if not self.data.select_dtypes(include="number").shape[1] == self.data.shape[1]:
            raise DataValidationError("Non-numeric values found.")
    
    def rename_columns(self):
        """
        Prepare data for alignment: 
        - Rename columns to distinguish from the data to be aligned with
        - Skip the x-axis, which will be used for alignment
        - Save a list of column (function) names for later reference
        """
        # cycle through all columns and create a dictionary with {old:new} column names
        # then rename all at once and save the new column names in a list
        rename_dict = {}
        i=1
        for column in self.data.columns:
            if column == self.Xaxis:
                continue
            rename_dict[column] = self.designator.format(i=i)
            i += 1
        self.data = self.data.rename(columns=rename_dict)
        self.function_names = list(rename_dict.values())


    def align_with(self, mergewith: "DataHandler"):
        """
        Align data with data of another DataHandler. There must be a common x-axis to align with.
        If 1:1 alignment is not possible or data is lost, DataAlignmentError is raised.
        """
        if self.Xaxis != mergewith.Xaxis:
            raise DataAlignmentError("Inconsistent x-Axis for Alignment.")
        try:
            aligned_data = self.data.merge(right=mergewith.data, on=self.Xaxis, how="inner", validate="1:1")
        except pd.errors.MergeError as e:
            raise DataAlignmentError("Error during 1:1 alignment.") from e
        if len(aligned_data) < len(self.data) or len(aligned_data) < len(mergewith.data):
            raise DataAlignmentError("Data was lost during Alignment.")
        return aligned_data

class TrainDataHandler (DataHandler):
    """
    TrainDataHandler Subclass:
    Holds methods to load from csv, 
    validate, rename and align (merge) data in form of
    x|y|y|y|y (x-values and 4 Training Functions)
    """
    def __init__(self, filepath: str, xaxis:str="x", designator:str="Y{i} (training func)"):
        super().__init__(filepath, xaxis, designator)
    def validate(self):
        super().validate()
        # Training Functions may not have duplicate x-values, must have 5 columns
        if self.data[self.Xaxis].duplicated().any():
            raise DataValidationError("x values are duplicate")
        if len(self.data.columns) != 5:
            raise DataValidationError("Wrong number of columns in Training Data")

class IdealDataHandler (DataHandler):
    """
    IdealDataHandler Subclass:
    Holds methods to load from csv, 
    validate, rename and align (merge) data in form of
    x|y|y|...|y (x values and 50 Ideal Functions)
    """
    def __init__(self, filepath: str, xaxis:str="x", designator:str="Y{i} (ideal func)"):
        super().__init__(filepath, xaxis, designator)
    def validate(self):
        super().validate()
        # Training Functions may not have duplicate x-values, must have 51 columns
        if self.data[self.Xaxis].duplicated().any():
            raise DataValidationError("x values are duplicate")
        if len(self.data.columns) != 51:
            raise DataValidationError("Wrong number of columns in Ideal Data")

class TestDataHandler (DataHandler):
    """
    TestDataHandler Subclass:
    Holds methods to load from csv, 
    validate and rename data in form of
    x|y (x-y Pairs or Points)
    """
    def __init__(self, filepath: str):
        # TestData columns will not be renamed and cannot choose x-axis / designator arguments.
        super().__init__(filepath, "x", "y")
    def validate(self):
        super().validate()
        # TestData must have excatly one x- and one y-column. 
        if len(self.data.columns) != 2:
            raise DataValidationError("Wrong number of columns in Test Data")
        if self.data.columns[0] != "x":
            raise DataValidationError("Wrong naming of columns in Test Data")
        if self.data.columns[1] != "y":
            raise DataValidationError("Wrong naming of columns in Test Data")
    def rename_columns(self):
        # TestData will not be renamed. 
        pass
