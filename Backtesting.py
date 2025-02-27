import numpy as np

def split_sequence(timeseries, n_steps):
    X, y = list(), list()
    for i in range(len(timeseries)):
        end_ix = i + n_steps
        if end_ix > len(timeseries) - 1:
            break
        seq_x, seq_y = timeseries[i:end_ix], timeseries[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def backtest_strategy(timeseries, n_steps, quantilles_decs, VaR_model, ES_model=None):
    # Split the time series into sequences
    X, y = split_sequence(timeseries, n_steps)

    # Initialize metrics
    VaR_exceedance = 0
    MAE = 0
    RMSE = 0
    Avg_ES = 0

    for i in range(len(X)):
        # Predict the VaR
        VaR = VaR_model.predict(X[i].reshape(1, -1))[0]  
        
        if y[i] < VaR:
            VaR_exceedance += 1
            # Predict the ES only if model is provided
            if ES_model is not None:
                ES = ES_model.predict(X[i].reshape(1, -1))[0]
                Avg_ES += ES
            # Update the MAE and RMSE
            MAE += abs(y[i] - VaR)
            RMSE += (y[i] - VaR) ** 2

    # Avoid division by zero
    if VaR_exceedance > 0:
        MAE /= VaR_exceedance
        RMSE = np.sqrt(RMSE / VaR_exceedance)
        if ES_model is not None:
            Avg_ES /= VaR_exceedance
    else:
        MAE = 0
        RMSE = 0
        Avg_ES = 0

    return VaR_exceedance, MAE, RMSE, Avg_ES


    