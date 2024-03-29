def DailyPrice2MonthlyReturn(df):
    '''
    Convert daily prices dataset into monthly return
    INPUT:
        df: daily dataset of stock prices to convert
    OUTPUT:
        dfM: monthly dataset of stock returns obtained from df
    '''
    from numpy import min, max
    from pandas import to_datetime, DataFrame

    #----- Create the index of the monthly dataset
    #Compute the years range of df
    index_year = df.index.year
    y_min, y_max = min(index_year), max(index_year)
    #Create the index of the monthly dataset
    months = list()
    for year in range(y_min, y_max+1):
        for month in range(1, 13):
            months.append( to_datetime(f'01/{month}/{year}',dayfirst=True) )
    #Adjust the index set by adding the last value (for temporal coherence)
    m_end = max(df[index_year==y_max].index.month)
    if m_end==12:
        months.append( to_datetime(f'01/01/{y_max+1}', dayfirst=True) )
    else:
        for _ in range(12, m_end+1, -1):
            months.pop(-1)
    #Adjust the index set by removing the first empty observations
    m_start = min(df[index_year==y_min].index.month)-1
    months = months[m_start:]

    #----- Create the monthly dataset
    dfM = DataFrame( index=months[1:], columns=df.columns )
    #Compute the returns
    for start, end in zip(months[:-1], months[1:]):
        temp = df[ (df.index>=start) & (df.index<end) ]
        dfM.loc[end] = (temp.iloc[-1] - temp.iloc[0]) / temp.iloc[0]

    return dfM

def DailyPrice2QuarterlyReturn(df):
    '''
    Convert daily prices dataset into quarterly return
    INPUT:
        df: daily dataset of stock prices to convert
    OUTPUT:
        dfM: quarterly dataset of stock returns obtained from df
    '''
    from numpy import min, max
    from pandas import to_datetime, DataFrame

    #----- Create the index of the quarterly dataset
    #Compute the years range of df
    index_year = df.index.year
    y_min, y_max = min(index_year), max(index_year)
    #Create the index of the quarterly dataset
    quarters = list()
    for year in range(y_min, y_max+1):
        for quarter in range(4):
            quarters.append( to_datetime(f'01/{quarter * 3 + 1}/{year}',dayfirst=True) )
    #Adjust the index set by adding the last value (for temporal coherence)
    q_end = max(df[index_year==y_max].index.month)
    if q_end == 12:
        quarters.append( to_datetime(f'01/01/{y_max+1}', dayfirst=True) )
    else:
        for _ in range(4, int(q_end/3), -1):
            quarters.pop(-1)
    q_start = min(df[index_year==y_min].index.month)
    if q_start != 1:
        for _ in range(0, int((q_start + 2) / 3), 1):
            quarters.pop(0)
    #Adjust the index set by removing the first empty observations

    #----- Create the quarterly dataset
    dfQ = DataFrame( index=quarters[1:], columns=df.columns )
    #Compute the returns
    for start, end in zip(quarters[:-1], quarters[1:]):
        temp = df[ (df.index>=start) & (df.index<end) ]
        dfQ.loc[end] = (temp.iloc[-1] - temp.iloc[0]) / temp.iloc[0]

    return dfQ

def DailyPrice2DailyReturn(df):
    '''
    Convert daily prices dataset into daily return
    INPUT:
        df: daily dataset of stock prices to convert
    OUTPUT:
        dfD: daily dataset of stock returns obtained from df
    '''

    dfD = df / df.shift(1) - 1
    dfD = dfD[1:]
    return dfD


def Return2Value(df, initial_capital):
    '''
    Convert daily returns series into values
    INPUT:
        df: series, returns to convert
        initial_capital: int, initial value of the series
    OUTPUT:
        dfV: series, values obtained from df returns
    '''
    dfV = df.copy()
    dfV[0] = initial_capital
    for i in range(1, len(dfV)):
        dfV[i] = dfV[i-1]*(1+dfV[i])
    return dfV
    
    
def l1_norm(x):
    '''
    Compute l1 norm of the input vector
    INPUT:
        x: list, vector
    OUTPUT:
        norm: int, l1 norm of x
    '''
    from numpy import abs, sum

    norm = sum(abs(x))
    return norm
    
    
def Mean_Variance(weights, assets):
    '''
    Compute the Mean-Variance coefficient of the portfolio
    INPUT:
        weights: array, weights of the portfolio
        assets: array, assets values
    OUTPUT:
        mean_var: int, opposite of Mean-Variance (to be minimized)
    '''
    from numpy import array, dot, mean
    
    #Compute mean
    mean = dot(mean(assets, axis=0), weights.T)
    #Estimate portfolio variance
    var = Var_Port(weights, assets)
    mean_var = -mean/var
    return mean_var


def Var_Port(weights, assets):
    '''
    Compute portfolio variance
    INPUT:
        weights: array, weights of the portfolio
        assets: array, assets values
    OUTPUT:
        tot_var: variance of the portfolio
    '''
    from numpy import array, cov

    cov_matr = cov(assets, rowvar=False)
    tot_var = 0
    for j in range(assets.shape[1]):
        tot_var += (weights[j]**2) * cov_matr[j, j]
        for k in range(j, assets.shape[1]):
            tot_var += 2 * weights[j] * weights[k] * cov_matr[j, k]
    return tot_var


def Expected_Sharpe(weights, alphas, resid):
    '''
    Expected Sharpe Ratio of the portfolio
    INPUT:
        weights: array, weights of the portfolio
        alphas: array, alpha of the assets in the portfolio
        resid: array, residual of the assets in the portfolio
    OUTPUT:
        sharpe: int, opposite of expected Sharpe Ratio (to be minimized)
    '''
    from numpy import array, dot, sqrt
    
    #Compute expected alpha
    expected_alpha = dot(weights, alphas)
    #If alphas is expected to be negative, consider inverse portfolio
    if expected_alpha < 0:
        weights = -weights
        expected_alpha = -expected_alpha
    #Estimate portfolio variance
    port_cov = Var_Port(weights, resid)
    #Compute Sharpe index on annual basis
    mu = (1+expected_alpha)**252 - 1
    deviation = sqrt(252*port_cov)
    sharpe = -mu/deviation
    return sharpe


def PooledRegression(Y, X, n_windows):
    '''
    Pooled Regression
    INPUT:
        Y: array, target values
        X: array, features matrix
        n_windows: int, number of windows
    OUTPUT:
        params: list, final params of the regression
    '''
    from numpy import array, mean
    from statsmodels.regression.linear_model import OLS

    #Compute windows length
    length = len(Y)//n_windows
    #Compute params in the other windows
    params = list()
    for t in range(n_windows-1):
        Y_temp = Y[t*length:(t+1)*length]
        X_temp = X[t*length:(t+1)*length]
        mdl = OLS(Y_temp, X_temp).fit()
        params.append(mdl.params)
    #Compute params in the last window
    Y_temp = Y[(n_windows-1)*length:]
    X_temp = X[(n_windows-1)*length:]
    mdl = OLS(Y_temp, X_temp).fit()
    params.append(mdl.params)
    #Return mean values
    params = mean(params, axis=0)
    return params


def PooledEnsembleRegressions(Y, X, n_windows):
    '''
    Pooled Ensemble Regression
    INPUT:
        Y: array, target values
        X: array, features matrix
        n_windows: int, number of windows
    OUTPUT:
        params: list, final params of the regression
    '''
    from numpy import array, mean, add
    from statsmodels.regression.linear_model import OLS
    from sklearn.svm import SVR

    #Compute windows length
    length = len(Y)//n_windows
    #Compute params in the other windows
    params = list()
    for t in range(n_windows-1):
        Y_temp = Y[t*length:(t+1)*length]
        X_temp = X[t*length:(t+1)*length]
        OLS_mdl = OLS(Y_temp, X_temp).fit()
        SVR_mdl = SVR(kernel='linear').fit(X_temp, Y_temp.ravel())
        mdl_params = add(SVR_mdl.coef_, OLS_mdl.params) / 2
        params.append(mdl_params)
    #Compute params in the last window
    Y_temp = Y[(n_windows-1)*length:]
    X_temp = X[(n_windows-1)*length:]
    OLS_mdl = OLS(Y_temp, X_temp).fit()
    SVR_mdl = SVR(kernel='linear').fit(X_temp, Y_temp.ravel())
    mdl_params = add(SVR_mdl.coef_, OLS_mdl.params) / 2
    params.append(mdl_params)
    #Return mean values
    params = mean(params, axis=0)
    params = params.flatten()
    return params

def PooledSVRRegressions(Y, X, n_windows):
    '''
    Pooled Ensemble Regression
    INPUT:
        Y: array, target values
        X: array, features matrix
        n_windows: int, number of windows
    OUTPUT:
        params: list, final params of the regression
    '''
    from numpy import array, mean, add
    from sklearn.svm import SVR

    #Compute windows length
    length = len(Y)//n_windows
    #Compute params in the other windows
    params = list()
    for t in range(n_windows-1):
        Y_temp = Y[t*length:(t+1)*length]
        X_temp = X[t*length:(t+1)*length]
        SVR_mdl = SVR(kernel='linear').fit(X_temp, Y_temp.ravel())
        params.append(SVR_mdl.coef_)
    #Compute params in the last window
    Y_temp = Y[(n_windows-1)*length:]
    X_temp = X[(n_windows-1)*length:]
    SVR_mdl = SVR(kernel='linear').fit(X_temp, Y_temp.ravel())
    params.append(SVR_mdl.coef_)
    #Return mean values
    params = mean(params, axis=0)
    params = params.flatten()
    return params