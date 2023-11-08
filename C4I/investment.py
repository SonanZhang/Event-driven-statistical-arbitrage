from C4I.utils import l1_norm, PooledRegression, Return2Value, PooledEnsembleRegressions, PooledSVRRegressions


def MarketNeutral_Portfolio(XD, XTrain, cluster, target_function,
                            windows_number=10, scaler='Standard',
                            opt_hyper={'maxiter':200, 'disp':True}, Ensemble = False):
    '''
    Create optimal market-neutral within a give cluster
    INPUT:
        XD: DataFrame, asset returns 
        XTrain: array,  2d array containing scaled features.
        cluster: list, cluster to work on, made up by list containing the index
            of considered risk factors and list containing the considered
            assets name
        target_function: function, target to optimize
        windows_number: int, number of windows for the pooled regression.
            Default=10
        scaler: str representing the scaler to use, either 'Standard' or
            'MinMax'. Other values will result in no scaling. Default='Standard'
        opt_hyper: dcit, hyperparameters for the minimizer, containing two keys:
            'maxiter' and 'disp'. Default={'maxiter':200, 'disp':True}
    OUTPUT:
        selected_portfolios: list, containing weights of the optimal portfolio
    '''
    from numpy import array, concatenate, dot
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from scipy.optimize import (minimize, Bounds,
                                LinearConstraint, NonlinearConstraint)
    
    #Define features matrix
    ClusterTrainSet = array([1]*XTrain.shape[0], ndmin=2).T
    for risk_factor in cluster[0]:
        ClusterTrainSet = concatenate([ClusterTrainSet,
                                        XTrain[:,risk_factor:risk_factor+1]],
                                      axis=1)
    #Preparing optimization: define constraints, alpha and epsilon vectors
    constraints = []
    alpha = []
    resid = []
    for asset in cluster[1]:
        #Define target and residual vectors
        Y_local = XD[asset].values
        if scaler == 'Standard':
            YTrain = StandardScaler().fit_transform(Y_local.reshape(-1,1))
        elif scaler == 'MinMax':
            YTrain = MinMaxScaler().fit_transform(Y_local.reshape(-1,1))
        else:
            YTrain = Y_local.reshape(-1,1)
        asset_res = []
        #Compute pooled regression and save coefficients and residuals
        # coeff = PooledRegression(YTrain, ClusterTrainSet, windows_number)
        if Ensemble:
            coeff = PooledEnsembleRegressions(YTrain, ClusterTrainSet, windows_number)
        else:
            coeff = PooledRegression(YTrain, ClusterTrainSet, windows_number)
        constraints.append( coeff[1:] ); alpha.append( coeff[0] )
        for t in range(len(YTrain)):
            temp = YTrain[t] - dot(ClusterTrainSet[t], coeff.T)
            asset_res.append(temp)
        resid.append(asset_res)
    constraints = array(constraints).T
    resid = array(resid)
    resid = resid.reshape(resid.shape[0], resid.shape[1]).T
    #Preparing optimization: define constraints, domain and starting point
    constr_lin = LinearConstraint(constraints, 0, 0)
    constr_non = NonlinearConstraint(l1_norm, 1, 1)
    domain = Bounds(-1, 1)
    w0 = [1/len(cluster[1])] * len(cluster[1])
    port_weights = minimize(target_function, w0, method="SLSQP", bounds=domain,
                            options=opt_hyper,
                            constraints=[constr_lin, constr_non],
                            args=(alpha, resid))
    #Save the results
    if port_weights.success:
        port_weights = port_weights.x
        target_value = target_function(port_weights, alpha, resid)
        #Check if the weights need to be inverted
        if target_value > 0:
            port_weights = -port_weights
        res = [target_value, port_weights]
    else:
        res = None
    return res


def Create_Portfolios(XD, Exog, clusters, target_function, windows_number=10,
                      scaler='Standard', opt_hyper={'maxiter':200, 'disp':True},
                      port_to_select=3):
    '''
    Create and select best market-neutral portfolios for the investment
    INPUT:
        XD: DataFrame, asset returns 
        Exog: array,  2d array containing features.
        clusters: dict containing two keys, 'alpha' and 'tau', and their grid of
            admissible values
        target_function: function, target to optimize
        windows_number: int, number of windows for the pooled regression.
            Default=10
        scaler: str representing the scaler to use, either 'Standard' or
            'MinMax'. Other values will result in no scaling. Default='Standard'
        opt_hyper: dcit, hyperparameters for the minimizer, containing two keys:
            'maxiter' and 'disp'. Default={'maxiter':200, 'disp':True}
        port_to_select: int, number of portfolios to select. Default=3
    OUTPUT:
        selected_portfolios: list, containing weights of the optimal portfolio
    '''
    from numpy import array, argmin, min
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    #Initialize results list
    target_result = list()
    #Create saved risk factors matrix
    if scaler == 'Standard':
        XTrain = StandardScaler().fit_transform(Exog)
    elif scaler == 'MinMax':
        XTrain = MinMaxScaler().fit_transform(Exog)
    else:
        XTrain = Exog.copy()

    #For each cluster, create market-neutral portfolio
    for cluster in clusters:
        cluster_result = MarketNeutral_Portfolio(XD=XD, XTrain=XTrain,
                                                 cluster=cluster, scaler=scaler,
                                                 target_function=target_function,
                                                 windows_number=windows_number,
                                                 opt_hyper=opt_hyper)
        if cluster_result != None:
            target_result.append( [cluster[1], cluster_result] )
    
    #Select and save the port_to_select optimal ones
    selected_portfolios = list()
    for _ in range(min([port_to_select, len(target_result)])):
        min_val = argmin( [portfolio[1][0] for portfolio in target_result] )
        temp = target_result.pop(min_val)
        temp[1] = temp[1][1]
        selected_portfolios.append( temp )
    return selected_portfolios


def Investment(X_Test, target_portfolios, n_round=3, initial_capital=1000):
    '''
    Compute the investment value during the test period
    INPUT:
        X_Test: DataFrame, asset returns in the test set
        target_portfolios: list, portfolios assets and weights
        n_round: int, number of decimals considered (if <=0, no round).Default=3
        initial_capital: int, initial value of the portfolio. Default=1000
    OUTPUT:
        port_test: series, values of the portfolio in the test set
    '''
    from pandas import DataFrame, Series
    from numpy import abs, array, dot, round, sum, zeros
    #Define test series
    port_test = zeros( len(X_Test.index) )
    #Add, portfolio by portfolio, the returns in the test set
    for port in target_portfolios:
        weights = port[1]/len(target_portfolios)
        #Check if weights have to be rounded
        if n_round > 0:
            weights = round( weights, n_round )
        for n_col, col in enumerate(port[0]):
            port_test += X_Test[col] * weights[n_col]
    #From returns to values
    port_test = Series(Return2Value(port_test, initial_capital),
                          index=X_Test.index, name='Portfolio')
    return port_test



def to_reduce_cluster(X_Test, clusters, reduce, earnings_by_tickers, pre_period_by_ticker, post_period_by_ticker):
    from pandas import DataFrame, Series
    from numpy import abs, array, dot, round, sum, zeros
    import pandas as pd
    import numpy as np 

    earning_weight_df = pd.DataFrame()
    earning_weight_df.index = X_Test.index
    cluster_weight_list = [1] * len(X_Test.index)
    earning_weight_df['weight'] = cluster_weight_list

    for i in range(len(clusters)):
        ticker = clusters[i]
        earning_dates = earnings_by_tickers[ticker]
        if (ticker not in pre_period_by_ticker.keys()):
            continue
        if (ticker not in post_period_by_ticker.keys()):
            continue 
        before_earnings_period = pre_period_by_ticker[ticker]
        after_earning_period = post_period_by_ticker[ticker]

        for date in earning_weight_df.index:
            normal_flag = True
            for earning_date_str in earning_dates:
                earning_date = pd.to_datetime(earning_date_str)
                diff_days = np.busday_count(date.strftime(format="%Y-%m-%d"), earning_date.strftime(format="%Y-%m-%d"))
                if diff_days >= 0 and diff_days < before_earnings_period:
                    normal_flag = False
                elif diff_days < 0 and diff_days > -after_earning_period:
                    normal_flag = False    
            if  not normal_flag:
                earning_weight_df.loc[date, 'weight'] = reduce
    return earning_weight_df

def InvestmnetWithEarningFilter(X_Test, target_portfolios, earning_by_tickers, pre_period_by_ticker, post_period_by_ticker, reduce_earning = 0.5, n_round=3, initial_capital=1000, earning_optimized=True):
    '''
    Compute the investment value during the test period and Filter with earning period filter
    INPUT:
        X_Test: DataFrame, asset returns in the test set
        target_portfolios: list, portfolios assets and weights
        n_round: int, number of decimals considered (if <=0, no round).Default=3
        initial_capital: int, initial value of the portfolio. Default=1000
    OUTPUT:
        port_test: series, values of the portfolio in the test set
    '''
    from pandas import DataFrame, Series
    from numpy import abs, array, dot, round, sum, zeros
    from C4I.investment import to_reduce_cluster
    import pandas as pd
    #Define test series

   
    port_test = zeros( len(X_Test.index) )
    #Add, portfolio by portfolio, the returns in the test set
    for port in target_portfolios:
        if earning_optimized:
            earning_weight_df = to_reduce_cluster(X_Test, port[0], reduce_earning, earning_by_tickers, pre_period_by_ticker, post_period_by_ticker)
        else:
            earning_weight_df = pd.DataFrame()
        weights = port[1]/len(target_portfolios)
       
        #Check if weights have to be rounded
        if n_round > 0:
            weights = round( weights, n_round )
        
        if earning_optimized:
            for n_col, col in enumerate(port[0]):
                port_test += (X_Test[col] * earning_weight_df['weight']) * weights[n_col]
        else:
            for n_col, col in enumerate(port[0]):
                port_test += X_Test[col] * weights[n_col]

        #Reduce weight around earnings
        
    #From returns to values
    port_test = Series(Return2Value(port_test, initial_capital),
                          index=X_Test.index, name='Portfolio')
    return port_test
