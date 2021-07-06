"""
This notebook aims to predict if and when invoices will be paid.

We seek to understand the factors of successful collection efforts of Accounts Receivables, and predict how late their payments will be received. In future, this will be used for individual client invoices when there is enough data given. 
"""

import math
import numpy as np
import pandas as pd
from datetime import datetime
import datetime as dt
from datetime import timedelta 
from scipy import stats

# Suppress warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error


"""Data Paths"""
input_csv = '../WA_Fn-UseC_-Accounts-Receivable.csv'
mean_days_late = '../output/mean_days_late.png'
payment_count = '../output/payment_count.png'
output_date_path = '../output/predicted_date.txt'
sample_csv_path = '../sample_invoice.csv'

# Random generator seed
random_seed = 2


"""Inputs"""
customer = "0688-XNJRO" # input customerID here
invoice_date = "2013/02/25"
due_date = "2013/02/28"
invoice_amount = 55.90
num_of_past_invoices = 2 # Also called `repeatCust` 
disputed = 0
with open('../output/percentage_late.txt') as f:
    late = f.readlines()
late = np.round(np.float64(late) / 100, 3)  

# Model features
features=['customerID', 'InvoiceDate', 'DueDate', 'InvoiceAmount', 'Disputed', 'Late']



df_receivable = pd.read_csv(input_csv)
df_receivable['InvoiceDate']= pd.to_datetime(df_receivable.InvoiceDate)

"""Generate a Late boolean using the variable 'DaysLate'. An invoice is considered "Late" if it is more than 7 days overdue."""
df_receivable['Late'] = df_receivable['DaysLate'].apply(lambda x: 1 if x > 7 else 0)


# countlate is a rolling count of the amount of late payments generated for each customer.
df_receivable['countlate']=df_receivable.Late.eq(1).groupby(df_receivable.customerID).apply(
    lambda x : x.cumsum().shift().fillna(0)).astype(int)

temp = pd.DataFrame(df_receivable.groupby(['countryCode'], axis=0)['DaysLate'].mean().reset_index())


"""Add a repeatCust variable"""
df1 = df_receivable[df_receivable['DaysLate']>0].copy()
df2 = pd.DataFrame(df1.groupby(['customerID'], axis=0)['DaysLate'].count().reset_index())
df2.columns = (['customerID','repeatCust'])
df3 = pd.merge(df_receivable, df2, how='left', on='customerID')
df3['repeatCust'].fillna(0, inplace=True)
df_receivable = df3

def func_IA (x):
    if x > 60: return "b. more than 60"
    else: return "a. less than 60"
df_receivable['InvoiceAmount_bin'] = df_receivable['InvoiceAmount'].apply(func_IA)

temp = pd.DataFrame(df_receivable.groupby(['InvoiceAmount_bin'], axis=0)['DaysLate'].mean().reset_index())


"""Generate more features and map some of the categorical variables to integers
Map some of the categorical variables to integers. Generate more insights about a customer with the given data. 
"""

df_receivable['Disputed'] = df_receivable['Disputed'].map({'No':0,'Yes':1})
df_receivable['PaperlessBill'] = df_receivable['PaperlessBill'].map({'Paper': 0,'Electronic': 1})
df_receivable['InvoiceQuarter']= pd.to_datetime(df_receivable['InvoiceDate']).dt.quarter



"""Label customers with integers for processing in the model."""
labels = df_receivable['customerID'].astype('category').cat.categories.tolist()
replace_map_comp = {'customerID' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}

# replace the customerID with Integers
df_receivable.replace(replace_map_comp, inplace=True)



"""
Train model to predict if a payment will be late
"""

cat_feats = ['InvoiceAmount_bin']
final_data = pd.get_dummies(df_receivable,columns=cat_feats,drop_first=True)
final_data["DueDate"] = pd.to_datetime(final_data["DueDate"])
final_data["InvoiceDate"] = pd.to_datetime(final_data["InvoiceDate"])
final_data["DueDate"] = final_data["DueDate"].map(datetime.toordinal)
final_data["InvoiceDate"] = final_data["InvoiceDate"].map(datetime.toordinal)

params = {"n_estimators":50, "max_depth":4, "random_state": random_seed}

X = final_data[features]
y = final_data['DaysLate']

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = random_seed) # default test size = 0.25


# Gradient Boosting Regressor model
GBR_model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=2)

# Fit Model
GBR_model.fit(X_train, y_train)

y_pred = GBR_model.predict(X_test)

# Checking the accuracy
GBR_model_accuracy = GBR_model.score(X_train,y_train)*100
print("Accuracy: ", round(GBR_model_accuracy,2),'%')
print("MSE: ", round(mean_squared_error(y_test,y_pred), 2))



"""
Predicted late payments (from all customers)
"""

y = pd.concat([y_test,pd.DataFrame(y_pred)],axis=1) # Using GBR model
y.columns = ('Actual','Prediction')

def act_decile (x):
    if x == 0: return "a. 0 days"
    elif x <= 4: return "b. (0-4] days"
    elif x <= 7: return "c. (4-7] days"
    elif x <= 10: return "d. (7-10] days"
    elif x <= 14: return "e. (10-14] days"
#     else: return "f. > 14 days"
y['act_bin'] = y['Actual'].apply(act_decile)


# Mean days late

temp = pd.DataFrame(y.groupby('act_bin', axis=0)['Actual','Prediction'].mean().reset_index())
temp.index = temp['act_bin']
tempgraph = temp.plot(marker='o',figsize=(10,7))
tempgraph.set_title("Mean days late",fontsize = 24)
tempgraph.set_ylabel("Days")
fig1 = tempgraph.get_figure()
fig1.savefig(mean_days_late)
fig1.show()


# Payment Count

temp1 = pd.DataFrame(y.groupby('act_bin', axis=0)['Actual','Prediction'].count().reset_index())
temp1.index = temp['act_bin']
tempgraph1 = temp1.plot(marker='o',figsize=(10,7))
tempgraph1.set_title("Payment count", fontSize=24)
tempgraph1.set_ylabel("Count")
fig2 = tempgraph1.get_figure()
fig2.savefig(payment_count)
fig2.show()


# Days late can be negative for early payment
y_date = pd.concat([df_receivable["DueDate"],pd.DataFrame(y_pred)],axis=1) # Using GBR model
y_date.columns = ('Due Date','Predicted Days Late')
y_date = y_date.dropna()
y_date['Predicted Days Late'] = y_date['Predicted Days Late'].round(0).astype(int)
y_date['Due Date'] = pd.to_datetime(y_date['Due Date'])
y_date_pred = y_date
y_date_pred["Predicted Payment Date"] = y_date['Due Date'] + pd.to_timedelta(y_date["Predicted Days Late"], unit='D')
y_date_pred["Predicted Payment Date"] = y_date_pred["Predicted Payment Date"].dt.date



"""
Forecast invoice payment date of a company, given the following invoice information: 
customerID, invoice_date, due_date, invoice_amount, num_of_past_invoices, disputed, late

We treat "Late" here as the percentage of late payments, based on their payment history. It is taken from the output of `percent_late.py`
""" 

# Create new dataframe based on inputs
df_ori = pd.read_csv(input_csv)
input_variables = pd.DataFrame([[customer, invoice_date, due_date, invoice_amount, disputed, late]], dtype=float)
input_variables.columns = ['customerID', 'InvoiceDate', 'DueDate', 'InvoiceAmount', 'Disputed', 'Late']
input_variables.to_csv('../sample_invoice.csv', index=False)


# Read sample invoice
input = pd.read_csv(sample_csv_path)
input["DueDate"] = pd.to_datetime(input["DueDate"])
input["InvoiceDate"] = pd.to_datetime(input["InvoiceDate"])
input["DueDate"] = input["DueDate"].map(datetime.toordinal)
input["InvoiceDate"] = input["InvoiceDate"].map(datetime.toordinal)

customer = df_ori["customerID"].astype('category').cat.categories.get_loc(customer)

# Get the model's prediction
input = input.assign(customerID = customer)
prediction = GBR_model.predict(input).round(0).astype(int)
prediction = int(prediction)
predicted_date = datetime.fromordinal(input["DueDate"]).date()
predicted_date = (predicted_date + timedelta(days = prediction)).strftime('%Y-%m-%d')

output = "Predicted days late: " + str(prediction) + "\n" + "Predicted payment date: " + predicted_date
print(output, file=open(output_date_path, 'w'))