# Key info

Forecasted cashflow will go in the `Forecast` tab where the user can input a end_date to forecast their cashflow to. The outputs can go into `Forecast Cash` and `Forecast Summary`.

It is based on the <a href="https://www.cashanalytics.com/differences-direct-indirect-cash-forecasting/">Direct Method of Forecasting</a>, which uses formula rather than 
machine learning to forecast cashflow.

`direct-forecast-newdata.py` is the same as `With_new_data.ipynb` (but cleaned up). Please use the `.py` file. 
They use the data that BNP Paribas handed us, and output the forecasted cashflow at the indicated end date and saves it into `output/forecast_cashflow.txt`.

`Cashflow_forecasting.ipynb` was used for initial experimentation and is based on our own randomly-generated data `generated-data.xlsx` and has much more parameters than what BNP handed us. 
