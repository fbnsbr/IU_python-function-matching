from src.data_handler import (
    TrainDataHandler, IdealDataHandler, TestDataHandler)
from src.visualizer import Visualizer
from src.function_matcher import FunctionMatcher
from src.database_manager import DatabaseManager


def main():    
    """
    Runs the a) Training-to-Ideal-Function matching b) testing data mapping workflow.
    Load and validate data, rename columns, align, merge, start a) and b), save results
    in SQL database, plot the results.
    Data files to be read: 
    - ./data/train.csv
    - ./data/idea.csv
    - ./data/test.csv
    SQL file written:
    - ./data/database.db
    Visualization:
    - ./main.html
    Raises:
    FileError: If an csv input file or SQLite database file cannot be accessed
    DataValidationError: If an input dataset is empty or incorrectly formatted
    DataAlignmentError: If the training and ideal functions cannot be properly aligned
    """

    # initialize, load, validate and rename each dataset according to its own class rules
    train_data = TrainDataHandler(r".\data\train.csv")
    ideal_data = IdealDataHandler(r".\data\ideal.csv")
    test_data = TestDataHandler(r".\data\test.csv")

    for handler in [train_data, ideal_data, test_data]:
        handler.load()
        handler.validate()
        handler.rename_columns()

    # align train data and ideal functions in one dataframe
    aligned_data = train_data.align_with(ideal_data)

    # Perform Training-Data to Idea-Function Matching
    # result is saved in "matcher.match_results"
    matcher = FunctionMatcher(aligned_data,  train_data.function_names, ideal_data.function_names)
    matcher.search_all_ideals()

    # Match Test Data to chosen Ideal Functions
    # result is saved in "matcher.test_mapping"
    matcher.match_test_data(test_data.data)

    # Save all train data, ideal functions and all results in SQL database
    database = DatabaseManager()
    database.save_dataframe(train_data.data, "Training_Data")
    database.save_dataframe(ideal_data.data, "Training_Functions")
    database.save_dataframe(matcher.test_mapping, "Test_Data_Mapping")

    # Perform Visualization of all results
    vis = Visualizer(aligned_data, matcher.match_results, matcher.test_mapping)
    vis.plot_training_results()
    vis.plot_test_mapping(matcher.test_mapping)
    vis.show_plots()

    print("\nTraining Data results")
    print(matcher.match_results)
    print("\nTest Data results")
    print(matcher.test_mapping.to_string())

if (__name__ == "__main__"):
    # if this is the main application, start main()
    main()
