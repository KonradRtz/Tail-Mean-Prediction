def split_sequence(timeseries, n_steps):
    X, y = list(), list()
    for i in range(len(timeseries)):
        end_ix = i + n_steps
        if end_ix > len(timeseries)-1:
            break
        seq_x, seq_y = timeseries[i:end_ix], timeseries[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def backtest_strategy(targetcolumn, timeseries, n_steps,quantilles_decs, VaR_model, ES_model=None):
    # Split the time series into sequences
    X, y = split_sequence(timeseries, n_steps)
    # Initialize the strategy
    VaR_exceedance = 0
    MAE = 0
    RMSE = 0
    Avg_ES = 0
    for i in range(len(X)):
        # Predict the VaR
        VaR = VaR_models[i].predict(X[i].reshape(1, -1))[0]
        if y[i] < VaR:
            VaR_exceedance += 1
            # Predict the ES
            ES = ES_model.predict(X[i].reshape(1, -1))[0]
            Avg_ES += ES
            # Update the MAE and RMSE
            MAE += abs(y[i] - VaR)
            RMSE += (y[i] - VaR)**2
    # Compute the metrics
    MAE = MAE / VaR_exceedance
    RMSE = np.sqrt(RMSE / VaR_exceedance)
    Avg_ES = Avg_ES / VaR_exceedance

    return VaR_exceedance, MAE, RMSE, Avg_ES

    