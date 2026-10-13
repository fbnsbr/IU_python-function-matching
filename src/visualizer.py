
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot
import pandas as pd
import numpy as np


class Visualizer:
    """
    Visualizer Class holds methods to plot training results 
    and test mapping, and to create a suitable layout.
    """
    def __init__(self, aligned_data:pd.DataFrame, training_results:pd.DataFrame, test_results:pd.DataFrame, 
                 scatter_color:str="blue", line_color:str="red", area_color:str="pink"):
        """
        Initialize Visualizer with
        - Aligned Data (contains all Training and Ideal Functions)
        - Training Results (contains the four Training Functions and their matched Ideal Functions)
        - Test Results (contains the mapping of x,y-testing data to the four Ideal Functions)
        - Scatter, Line and Area Color may be adjusted
        """
        self.aligned_data = aligned_data
        self.training_results = training_results
        self.test_results = test_results
        if self.aligned_data.empty or self.training_results.empty or self.test_results.empty:
            raise ValueError("No valid visualization data given, run calculations first.")
        self.scatter_color = scatter_color
        self.line_color = line_color
        self.area_color = area_color
        self.plots = []

    def show_plots(self):
        """
        Show the collected plots in one suitable layout.
        """
        if not self.plots: 
            return
        layout = gridplot(self.plots, ncols=2)
        show(layout)
        
    def plot_training_results(self):
        """
        Plot Training Results:
        Plots each training function (scatter) with chosen ideal function (line)
        """
        # Figures 1-4: Iterate through training and ideal functions
        for train_f, ideal_f in zip(self.training_results["Train_f"],self.training_results["Ideal_f"]):
            p =  figure(
                title=f"{train_f}[{self.scatter_color}] vs {ideal_f}[{self.line_color}]",
                x_axis_label="x",
                y_axis_label="y",
                active_scroll="wheel_zoom",
                active_drag="pan")
            # get the corresponding train + ideal columns (function y-values) from aligned_data and plot
            ideal_x = self.aligned_data["x"].tolist()
            train_y = self.aligned_data[train_f]
            ideal_y = self.aligned_data[ideal_f]
            p.scatter(ideal_x, train_y, color=self.scatter_color, line_width=1, legend_label=train_f)
            p.line(ideal_x, ideal_y, color=self.line_color, line_width=1, legend_label=ideal_f)
            self.plots.append(p)

    def plot_test_mapping(self, testmapping: pd.DataFrame):
        """
        Plot Test Mapping:
        Plots each of the four chosen Ideal Functions with their mapped x,y-Points
        Plots one figure with unmapped x,y-Points
        """
        # Figures 5-8: plot mapped testing data: iterate trough the four ideal functions
        for ideal_f in self.training_results["Ideal_f"]:
            p =  figure(
                title=f"Mapped Testing Data [{self.scatter_color}] to {ideal_f}[{self.line_color}]",
                x_axis_label="x",
                y_axis_label="y",
                active_scroll="wheel_zoom",
                active_drag="pan")
            # plot ideal function with acceptance margin (+/- threshold)
            ideal_x = self.aligned_data["x"].tolist()
            ideal_y = self.aligned_data[ideal_f]
            threshold = self.training_results.loc[
                self.training_results["Ideal_f"]==ideal_f, "MaxDev"].to_numpy() * np.sqrt(2)
            p.varea(x=ideal_x, y1=ideal_y-threshold, y2=ideal_y+threshold, color=self.area_color, fill_alpha=0.5, legend_label="Acceptance Margin")
            p.line(x=ideal_x, y=ideal_y, color=self.line_color, line_width=1, legend_label=ideal_f)
            # find and plot corresponding x-y-points
            points = testmapping[testmapping["No_of_ideal_func"]==ideal_f]
            p.scatter(points["x"], points["y"], color=self.scatter_color, line_width=1, legend_label="Mapped Test Data")
            self.plots.append(p)

        # Figure 9: plot unmapped testing data
        p =  figure(
            title=f"Unmapped Testing Data [{self.scatter_color}] with Ideal Functions [{self.line_color}]",
            x_axis_label="x",
            y_axis_label="y",
            active_scroll="wheel_zoom",
            active_drag="pan")
        points = testmapping[testmapping["No_of_ideal_func"]=="None"]
        p.scatter(points["x"].tolist(), points["y"].tolist(), color=self.scatter_color, line_width=1, legend_label="Unmapped Test Data")
        # in this figure, plot all four ideal functions with acceptance margin (+/- threshold) for reference
        for ideal_f, linestyle in zip(self.training_results["Ideal_f"], ["solid", "dashed", "dotted", "dotdash"]):
                    ideal_x = self.aligned_data["x"].tolist()
                    ideal_y = self.aligned_data[ideal_f]
                    threshold = self.training_results.loc[
                         self.training_results["Ideal_f"]==ideal_f, "MaxDev"].to_numpy() * np.sqrt(2)
                    p.varea(x=ideal_x, y1=ideal_y-threshold, y2=ideal_y+threshold, color=self.area_color, fill_alpha=0.5, legend_label="Acceptance Margin")
                    p.line(x=ideal_x, y=ideal_y, color=self.line_color, line_width=1, line_dash=linestyle, legend_label=ideal_f)                  
        self.plots.append(p)

