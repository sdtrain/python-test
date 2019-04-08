#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import nsepy
import pandas as pd
import numpy as np
import datetime
from datetime import date, timedelta
from time import time
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib

# GET COMPANY STOCK DATA
def get_stock_data(company, start_date, end_date):
    return nsepy.get_history(symbol=company.upper(), start=start_date, end=end_date)

# GET PRICE DATA FROM COMPANY STOCK DATA
def get_price_features(data):
    day = [date.day for date in data.index.values]
    month = [date.month for date in data.index.values]
    prev_close_1 = data['Close'].values.reshape(1, -1)
    prev_close_2 = data.Close.shift(1)
    prev_close_2 = prev_close_2.fillna(0)
    prev_close_3 = data.Close.shift(2)
    prev_close_3 = prev_close_3.fillna(0)
    prev_close_4 = data.Close.shift(3)
    prev_close_4 = prev_close_4.fillna(0)
    prev_close_5 = data.Close.shift(4)
    prev_close_5 = prev_close_5.fillna(0)
    prev_close_6 = data.Close.shift(5)
    prev_close_6 = prev_close_6.fillna(0)
    prev_close_7 = data.Close.shift(6)
    prev_close_7 = prev_close_7.fillna(0)
    roll_avg_10 = data['Close'].rolling(10).mean()
    roll_avg_10 = roll_avg_10.fillna(0)
    price_X = np.dstack((day, month, prev_close_1, prev_close_2, prev_close_3, prev_close_4, prev_close_5, prev_close_6, prev_close_7, roll_avg_10))[0]
    price_X = price_X[-1,:].reshape(1, -1)
    
    return price_X

# GET VOLUME SHOCK DATA FROM COMPANY STOCK DATA
def get_volume_shock_features(data):
    day = [date.day for date in data.index.values]
    month = [date.month for date in data.index.values]    
    prev_volume_1 = data.Volume.shift(0)
    prev_volume_1 = prev_volume_1.fillna(0)
    prev_volume_2 = data.Volume.shift(1)
    prev_volume_2 = prev_volume_2.fillna(0)
    prev_volume_3 = data.Volume.shift(2)
    prev_volume_3 = prev_volume_3.fillna(0)
    prev_volume_4 = data.Volume.shift(3)
    prev_volume_4 = prev_volume_4.fillna(0)
    prev_volume_5 = data.Volume.shift(4)
    prev_volume_5 = prev_volume_5.fillna(0)
    prev_volume_6 = data.Volume.shift(5)
    prev_volume_6 = prev_volume_6.fillna(0)
    prev_volume_7 = data.Volume.shift(6)
    prev_volume_7 = prev_volume_7.fillna(0)
    vol_roll_avg_10 = data['Volume'].rolling(10).mean()
    vol_roll_avg_10 = vol_roll_avg_10.fillna(0)
    vol_X = np.dstack((day, month, prev_volume_1, prev_volume_2, prev_volume_3, prev_volume_4, prev_volume_5, prev_volume_6, prev_volume_7, vol_roll_avg_10))[0]
    vol_X = vol_X[-1,:].reshape(1, -1)
    
    return vol_X

# PRICE PREDICTION 
def predict_price(company, data):
    # Predict Stock Closing Price for Tomorrow
    model = joblib.load('r_{}_model.sav'.format(company.lower()))
    prediction = model.predict(data)
    
    return prediction

# VOLUME SHOCK PREDICTION 
def predict_volume_shock(company, data):
    # Predict Stock Volume Shock for Tomorrow
    model = joblib.load('gbc_{}_vol_model.sav'.format(company.lower()))
    prediction = model.predict(data)
    
    return prediction

if __name__ == '__main__':
    start_time_1 = time()
    company = sys.argv[1]
    company = company.upper()
    if (company == "TCS") or (company == "INFY"):
        print('Getting Stock Data for %s' % company)
    else:
        print("Wrong Input - Please choose from 'TCS' or 'INFY' only")
        quit()
    
    end_date = datetime.datetime.now().date()
    start_date = end_date - timedelta(30)
    data = get_stock_data(company, start_date, end_date)
    data = data.iloc[-11:]
    
    start_time_2 = time()
    print('Your predictions are being made...')
    price_X = get_price_features(data)
    volume_X = get_volume_shock_features(data)
    price_prediction = predict_price(company, price_X)
    volume_shock_prediction = predict_volume_shock(company, volume_X)
    
    print("Tomorrow's Predicted Closing Price is: {:.2f}".format(price_prediction[0]))
    
    if volume_shock_prediction == 1: print("There is a chance of Volume Shock tomorrow.")
    elif volume_shock_prediction == 0: print("There is less chance of Volume Shock tomorrow.")
    else: print("There is an error in Volume Shock Prediction.")
        
    end_time = time()
    print("Prediction Time --- %s ms ---" % ((end_time - start_time_2)*1000))
    print("Total Time --- %s ms ---" % ((end_time - start_time_1)*1000))

