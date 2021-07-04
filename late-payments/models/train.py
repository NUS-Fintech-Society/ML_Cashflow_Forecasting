"""
This notebook aims to predict if and when invoices will be paid.
Understand the factors of successful collection efforts of Accounts Receivables, and predict how late their payments will be received. In future, this will be used for individual client invoices when there is enough data given. 
"""

import math
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats

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
input_csv = './WA_Fn-UseC_-Accounts-Receivable.csv'
mean_days_late = 'output/mean_days_late.png'
payment_count = 'output/payment_count.png'
output_deviation = 'output/deviation.png'

# Random generator seed
random_seed = 4

# Model features
features=['countryCode', 'customerID', 'InvoiceAmount', 'Late', 'DaysToSettle', 'countlate']


df_receivable = pd.read_csv(input_csv)
df_receivable['InvoiceDate']= pd.to_datetime(df_receivable.InvoiceDate)

"""Generate a Late boolean using the variable 'DaysLate'. An invoice is considered "Late" if it is more than 7 days overdue."""

df_receivable['Late'] = df_receivable['DaysLate'].apply(lambda x: 1 if x > 7 else 0)

"""countlate is a rolling count of the amount of late payments generated for each customer."""

df_receivable['countlate']=df_receivable.Late.eq(1).groupby(df_receivable.customerID).apply(
    lambda x : x.cumsum().shift().fillna(0)).astype(int)

temp = pd.DataFrame(df_receivable.groupby(['countryCode'], axis=0)['DaysLate'].mean().reset_index())
plt.figure(figsize=(10,6))
sns.barplot(x="countryCode", y="DaysLate",data=temp)


"""Percentage of late payments of a customer"""

customer_late = pd.crosstab(index=df_receivable["customerID"], columns=df_receivable['Late'])

# customer = "0688-XNJRO" # input customerID here

# late_0 = customer_late[customer_late.index == customer][0]
# late_1 = customer_late[customer_late.index == customer][1]
# total = late_0 + late_1
# percentage_of_late_payment = (late_1 / total) * 100

# print("% of late payments \n" + percentage_of_late_payment.to_string() + "%")



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

plt.figure(figsize=(4,6))
sns.barplot(x="InvoiceAmount_bin", y="DaysLate",data=temp,color='blue')

"""Generate more features and map some of the categorical variables to integers
Map some of the categorical variables to integers. Generate more insights about a customer with the given data. 

Eg. Is a company more likely to pay on time if the order occurs at the end of the year? For that we use the **InvoiceQuarter** variable to look at the time that the invoice was made.
"""

df_receivable['Disputed'] = df_receivable['Disputed'].map({'No':0,'Yes':1})
df_receivable['PaperlessBill'] = df_receivable['PaperlessBill'].map({'Paper': 0,'Electronic': 1})
df_receivable['InvoiceQuarter']= pd.to_datetime(df_receivable['InvoiceDate']).dt.quarter

"""### Relations of "Late" to other variables"""

plt.figure(figsize=(10,8))
ax = sns.countplot(df_receivable['countryCode'],hue=df_receivable['Late'],palette="YlGn")

for p in ax.patches:
    txt = str((p.get_height()).round(1))
    txt_x = p.get_x()
    txt_y = p.get_height()
    ax.text(txt_x,txt_y,txt, fontSize=18)
    
ax.set_title("Late payments per country code",fontsize=24)



plt.figure(figsize=(10,8))
ax2 = sns.countplot(df_receivable['InvoiceQuarter'],hue=df_receivable['Late'],palette='bright')
ax2.set_title("Late payments per invoice quarter",fontsize=24)

"""### Distributions of Invoice Amounts and Days to settle
This can be helpful if we wish to know within reasonable assumptions what our confidence intervals are for payments or how long it takes for a customer to settle.
"""

plt.figure(figsize=(8,8))
invoice_plot = sns.distplot(df_receivable['InvoiceAmount'],color='green')
invoice_plot.axes.set_title("Invoice Amount",fontsize=24)

plt.figure(figsize=(8,8))
settle_plot = sns.distplot(df_receivable['DaysToSettle'],color='blue')
settle_plot.axes.set_title("Days to Settle",fontsize=24)

"""Label customers with integers for processing in the model."""
labels = df_receivable['customerID'].astype('category').cat.categories.tolist()
replace_map_comp = {'customerID' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}

# replace the customerID with Integers
df_receivable.replace(replace_map_comp, inplace=True)



"""Train model to predict if a payment will be late"""

cat_feats = ['InvoiceAmount_bin']
final_data = pd.get_dummies(df_receivable,columns=cat_feats,drop_first=True)

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
print(round(GBR_model_accuracy,2),'%')
print(mean_squared_error(y_test,y_pred))


"""Predicted late payments (from all customers)
In future, can narrow to payments from individual customers as data now is too little
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



# Model Deviation
# The following code errors in python3 but works for python2 in Jupyter Notebook :( 

"""
test_score = np.zeros(params['n_estimators'], dtype=np.float64)
for i, y_pred in enumerate(GBR_model.staged_predict(X_test)):
    test_score[i] = GBR_model.loss_(y_test, y_pred)

fig = plt.figure(figsize=(10, 7))
plt.subplot(1, 1, 1)
plt.title('Deviance', fontSize = 24)
plt.plot(np.arange(params['n_estimators']) + 1, GBR_model.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Loss')
plt.show()
"""