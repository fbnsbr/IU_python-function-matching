import unittest
import pandas as pd
from src.function_matcher import FunctionMatcher

class UnitTestFuncionMatcher(unittest.TestCase):
    """
    TestCase-derived Class to test FunctionMatcher Class
    """
    def setUp(self):
        """
        Set up the common requirements for each independent test
        create example aligned data and function names,
        create FunctionMatcher object
        """
        self.aligned_data = pd.DataFrame({
            "x":      [0,   1,   2],

            "Train1": [0,   1,   2],
            "Train2": [2,   2,   3],

            "Ideal1": [2,   2,   2],
            "Ideal2": [0,   0,   1],
            "Ideal3": [10,  10,  10]
        })
        self.train_names = ["Train1", "Train2"]
        self.ideal_names = ["Ideal1", "Ideal2", "Ideal3"]
        self.matcher = FunctionMatcher(self.aligned_data, 
                    self.train_names, self.ideal_names)

    def tearDown(self):
        return super().tearDown()

    def test_search_all_ideals(self):
        """
        Check if the Ideal Functions are correctly assigned to the Training Functions
        """
        self.matcher.search_all_ideals()
        self.assertEqual(len(self.matcher.match_results), 2)
        self.assertEqual(self.matcher.match_results.iloc[0]["Ideal_f"],"Ideal2")
        self.assertEqual(self.matcher.match_results.iloc[1]["Ideal_f"],"Ideal1")
        self.assertEqual(self.matcher.matched_ideals_list,["Ideal2", "Ideal1"])


    def test_match_ideal_to_train(self):
        """
        Check if Ideal2 Function is correctly assigned to the Train1 Function
        """
        ideal_name, squared_sum, max_dev = (self.matcher.match_ideal_to_train("Train1"))
        self.assertEqual(ideal_name, "Ideal2")
        self.assertEqual(squared_sum, 2)
        self.assertEqual(max_dev, 1)

    def test_compare_functions(self):
        """
        Check if the Function comparison delivers correct results
        """
        squared_sum, max_dev = self.matcher.compare_functions("Train1", "Ideal1")
        self.assertEqual(squared_sum, 5)
        self.assertEqual(max_dev, 2)


    def test_match_test_data_valid_points(self):
        """
        Check if two example x,y-Points are correctly mapped to Ideal1/Ideal2 
        """
        self.matcher.search_all_ideals()
        test_data = pd.DataFrame({
            "x": [0, 2],
            "y": [2, 1] })
        self.matcher.match_test_data(test_data)
        self.assertEqual( 
            self.matcher.test_mapping.iloc[0]["No_of_ideal_func"], "Ideal1")
        self.assertEqual(
            self.matcher.test_mapping.iloc[1]["No_of_ideal_func"], "Ideal2")


    def test_match_test_data_invalid_point(self):
        """
        Check if one example x,y-Points are correctly left unmapped
        """
        self.matcher.search_all_ideals()
        test_data = pd.DataFrame({
            "x": [0],
            "y": [5]      })
        self.matcher.match_test_data(test_data)
        self.assertEqual(
            self.matcher.test_mapping.iloc[0]["No_of_ideal_func"], "None")

    def test_match_test_data_duplicate_x(self):
        """
        Check if two example x,y-Points with duplicate x-values are correctly handled
        """
        self.matcher.search_all_ideals()
        test_data = pd.DataFrame({
            "x": [0, 0],
            "y": [0, 10]
        })
        self.matcher.match_test_data(test_data)
        self.assertEqual(len(self.matcher.test_mapping), 2)
        self.assertEqual(
            self.matcher.test_mapping.iloc[0]["No_of_ideal_func"], "Ideal2")
        self.assertEqual(
            self.matcher.test_mapping.iloc[1]["No_of_ideal_func"], "None")
