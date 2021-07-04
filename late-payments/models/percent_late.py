"""
This notebook is a script to find the percentage of late payments of a customer / vendor. This will give the client an idea of which customers will pay the fastest based on their payment history, thereby improving the efficiency of payment collections.
"""

import math
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
from sklearn import metrics


"""Data Paths"""
input_csv = './WA_Fn-UseC_-Accounts-Receivable.csv'

"""Inputs"""
days_late = 7 # input number of days for a payment to be considered "late". Can be 7/30/90.
customer = "0688-XNJRO" # input customerID here


df_receivable = pd.read_csv(input_csv)
df_receivable['InvoiceDate']= pd.to_datetime(df_receivable.InvoiceDate)


"""Generate a Late boolean using the variable 'DaysLate'. An invoice is considered "Late" if it is more than 7 days overdue."""

df_receivable['Late'] = df_receivable['DaysLate'].apply(lambda x: 1 if x > days_late else 0)


"""Percentage of late payments of a customer"""

customer_late = pd.crosstab(index=df_receivable["customerID"], columns=df_receivable['Late'])

late_0 = customer_late[customer_late.index == customer][0]
late_1 = customer_late[customer_late.index == customer][1]
total = late_0 + late_1
percentage_of_late_payment = round((late_1 / total) * 100, 2)

output = "Percentage of late payments: \n" + percentage_of_late_payment.to_string() + "%"
print(output, file=open('output/percent_late.txt', 'w'))
