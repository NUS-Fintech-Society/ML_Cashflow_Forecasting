import pandas as pd
import random
import datetime
import numpy as np

# Matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
# %matplotlib inline
import matplotlib
import seaborn as sns
# Scipy helper functions
from scipy import stats
from random import randrange
from datetime import timedelta


"""Data Paths"""
input_xlsx = '../data/bnp_data.xlsx'
output_path = './output/forecast_cashflow.txt'

"""Inputs"""
start_date = "2010-01-02"
end_date = "2010-04-29"


df_data = pd.read_excel(input_xlsx)
df_data["Date"] = pd.to_datetime(df_data.Date)
df_data.sort_values(by=['Date'], inplace=True, ascending=True)

# Sum net cash (per day)
df_data_sum = df_data.resample('D', on="Date").sum()
df_data_sum.head()


"""
Function to calculate daily forecasted cash flow, `forecastCashFlow`.

Formula: Forecasted cash flow = Forecasted inflow - Forecasted outflow
Optional parameter (`forecastedDate`) : For a user to input a target date to forecast to. 
"""

def forecastCashFlow(df_data, forecastedDate, startdate):
    forecastedDate_time = datetime.datetime.strptime(forecastedDate, '%Y-%m-%d')
    forecasted_inflow = 0
    forecasted_outflow = 0
    after_start_date = df_data["Date"] >= startdate
    before_end_date = df_data["Date"] <= forecastedDate_time
    between_two_dates = after_start_date & before_end_date
    filtered_dates = df_data.loc[between_two_dates]
    for i in range (filtered_dates.shape[0]):
       amount = filtered_dates.iloc[i, 3]
       forecasted_outflow += amount

    return forecasted_inflow - forecasted_outflow


# Return forecasted cash flow at the user-inputted date
output = forecastCashFlow(df_data, end_date, start_date)
print(output, file=open(output_path, 'w'))
