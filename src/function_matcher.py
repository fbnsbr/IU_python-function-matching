import numpy as np
import pandas as pd

class FunctionMatcher:
    """
    FunctionMatcher-Class holds methods for 
    - matching ideal functions to training functions,
    - mapping test-data (x,y-points) to the chosen ideal functions.
    Initialize with
    - aligned_data: x-axis, training functions and ideal functions in one pd.dataframe
    - lists of the names of training / ideal functions in that dataframe.
    Holds results in
    - match_results: results of the Training Data-to-Ideal Function matching 
    - test_mapping: results of the mapping of x,y-pairs to the selected Ideal Functions
    """

    def __init__(self, aligned_data:pd.DataFrame, train_f_names:list[str], ideal_f_names:list[str]):
        """
        Initialize with aligned_data (x-axis, training functions and ideal functions) 
        and lists of the names of training and ideal functions.
        """
        self.aligned_data = aligned_data
        self.train_f_names = train_f_names
        self.ideal_f_names = ideal_f_names
        self.match_results = pd.DataFrame(columns=("Train_f", "Ideal_f", "SumSquaredDev", "MaxDev"))
        self.matched_ideals_list=[]
        self.test_mapping = pd.DataFrame(columns=("x", "y", "Delta_Y", "No_of_ideal_func"))

    def search_all_ideals(self):
        """
        Match Ideal Functions to the given Training Functions.
        Returns a datafield containing each Train Function names, list of matched 
        Ideal Function names, Sum of Squared y-Deviations, and Maximum y-Deviation.
        """
        for train_f in self.train_f_names:
            ideal_name, squaredsum, maxdev = self.match_ideal_to_train(train_f)
            self.match_results.loc[len(self.match_results)] = [train_f, ideal_name, squaredsum, maxdev]
            self.matched_ideals_list.append(ideal_name)

    def match_ideal_to_train(self, train_name:str) -> tuple:
        """
        Match one Ideal Function to one Training Function. 
        Returns the name of the best matched Ideal Function,
        its Sum of Squared y-Deviations, and Maximum y-Deviation.
        """
        # variables to hold the most recent best fit ideal function during search
        best_squared_sum = float("inf")
        best_maxdev = 0
        best_name = ""
        
        for ideal_f in self.ideal_f_names:
            # iterate through all ideal functions and keep the best fit (least sum of squared deviations)
            squared_sum, max_deviation = self.compare_functions(compare1=train_name, compare2=ideal_f)
            if squared_sum < best_squared_sum:
                best_squared_sum = squared_sum
                best_maxdev = max_deviation
                best_name = ideal_f
        # all ideal functions checked, return the best fit
        return best_name, best_squared_sum, best_maxdev


    def compare_functions (self, compare1:str, compare2:str) -> tuple:
        """
        Compare two functions y-values. They must have already been aligned on x-values.
        Returns Sum of Squared y-Deviations and Maximum y-Deviation.
        """
        # Calculate absolute differences, return maximum deviation and sum of squared deviations
        abs_dif = (self.aligned_data[compare1] - self.aligned_data[compare2]).abs()
        max_deviation = abs_dif.max()
        squared_sum = (abs_dif**2).sum()
        return (squared_sum, max_deviation)

    def match_test_data (self, test_data: pd.DataFrame):
        """
        Map test points to eligible ideal functions.
        A point is eligible for an ideal function when its absolute deviation
        does not exceed the training function's maximum deviation multiplied
        by sqrt(2). Among eligible functions, the smallest absolute deviation
        is selected.
        """
        if self.match_results.empty:
            raise ValueError("Run search_all_ideals() before matching test data.")
       
        # calculate the acceptance thresholds (MaxDev * sqrt(2)) of the four selected ideal functions
        thresholds = self.match_results["MaxDev"].to_numpy(dtype=float) * np.sqrt(2)

        # prepare a table with "x" and the four selected ideal functions (x|y1|y2|y3|y4)
        matched_ideals = self.aligned_data[["x"] + self.matched_ideals_list]

        # collect resulting mappings in a list of dictionaries first and create dataframe later
        mapped_rows = []
        
        # read test-data (x,y) line by line
        for x,y in test_data[["x","y"]].itertuples(index=False, name=None):
            # for each x, select corresponding y-values of the four matched ideal functions.
            # remove x-column and flatten (y1,y2,y3,y4)
            ideal_y = matched_ideals[matched_ideals["x"] == x].drop(columns="x").to_numpy(dtype=float).flatten()
            # calculate each Delta_Y: the y-distance between point and each ideal function
            delta_y = np.abs((ideal_y - y))
            # compare each Delta_Y with its threshold to receive four bool values
            eligible = delta_y <= thresholds
            # if there are eligible ideal functions, choose the lowest DeltaY and corresponding function name
            if eligible.any():
                # collect the DeltaY of eligible functions. For non-eligible functions, set value to infinity
                # find the smallest DeltaY and its index
                # with that index, get the name of the ideal function and its delta_y
                eligible_deltas = np.where(eligible, delta_y, np.inf)
                best_index = np.argmin(eligible_deltas)
                best_ideal = self.matched_ideals_list[best_index]
                best_delta = delta_y[best_index]
                # save the mapping
                mapped_rows.append({
                    "x": x,
                    "y": y,
                    "Delta_Y": best_delta,
                    "No_of_ideal_func": best_ideal
                })
            else:
                # no eligible ideal function
                mapped_rows.append({
                    "x": x,
                    "y": y,
                    "Delta_Y": np.nan,
                    "No_of_ideal_func": "None"
                })

        # create the dataframe with mapping results 
        self.test_mapping = pd.DataFrame(mapped_rows)
                
                
            

