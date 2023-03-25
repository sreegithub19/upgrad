# Project Name
Assignment:

A US-based housing company named Surprise Housing has decided to enter the Australian market. The company uses data analytics to purchase houses at a price below their actual values and flip them on at a higher price. For the same purpose, the company has collected a data set from the sale of houses in Australia. The data is provided in the CSV file below.

The company is looking at prospective properties to buy to enter the market. You are required to build a regression model using regularisation in order to predict the actual value of the prospective properties and decide whether to invest in them or not.

The company wants to know: Which variables are significant in predicting the price of a house, and How well those variables describe the price of a house.

Also, determine the optimal value of lambda for ridge and lasso regression.

Business Goal:

You are required to model the price of houses with the available independent variables. This model will then be used by the management to understand how exactly the prices vary with the variables. They can accordingly manipulate the strategy of the firm and concentrate on areas that will yield high returns. Further, the model will be a good way for management to understand the pricing dynamics of a new market.


## Table of Contents
    Problem_statement
    Importing_required_libraries
    Loading_the_dataset
    Inspecting_the_dataset
    Exploratory_Data_Analysis
    Data_Preparation
    Handling_missing_values
    Feature_Engineering
    Ridge_regression
    Lasso_regression
    Final_regression_models
    Subjective_Questions_and_Answers


## General Information
- What is the dataset that is being used?
Dataset used: https://ml-course3-upgrad.s3.amazonaws.com/Assignment_+Advanced+Regression/train.csv

## Conclusions
- We are able to achieve an R^2 score of 0.82 approx on both Ridge and Lasso Models. The follwing factors influence the house price the most as demosntrated by both the models:-

        Total area in square foot
        Total Garage Area
        Total Rooms
        Overall Condition
        Lot Area
        Centrally Air Conditioned
        Total Porch Area (Open + Enclosed)
        Kitchen Quality
        Basement Quality 
- If we have too many variables and one of our primary goal is feature selection, then we will use Lasso.
- If we don't want to get too large coefficients and reduction of coefficient magnitude is one of our prime goals, then we will use Ridge Regression.




## Contact
Created by <a href="https://github.com/sreegithub19">sreegithub19</b> - feel free to contact me!
