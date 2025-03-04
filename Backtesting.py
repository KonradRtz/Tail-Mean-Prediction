import pandas as pd
import numpy as np 

def backtest(timeseries_df, target_col, VaR_model,ESModel = None):
    VaR_breaches = 0
    MAE = 0
    RMSE = 0
    X = timeseries_df.drop(target_col, axis = 1)
    X.drop(['Date','Name_idx'], axis = 1, inplace = True)
    y = timeseries_df[target_col]
    for i in range(len(timeseries_df)):
        y_pred = VaR_model.predict(X.iloc[i].values.reshape(1,-1))
        y_true = y.iloc[i]
        if y_true < y_pred:
            VaR_breaks += 1
            if ESModel is not None:
                ES = ESModel.predict(X.iloc[i].values.reshape(1,-1))
                MAE += abs(y_true - ES)
                RMSE += (y_true - ES)**2
    if VaR_breaks == 0:
        MAE = 0
        RMSE = 0
    else:
        MAE = MAE/VaR_breaches
        RMSE = np.sqrt(RMSE/VaR_breaches)
    exp_breaches = len(timeseries_df)*0.025
    return exp_breaches, VaR_breaks, MAE, RMSE


    
        
