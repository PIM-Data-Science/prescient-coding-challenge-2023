# %%
import numpy as np
import pandas as pd
import datetime
import plotly.express as px


print('---Python script Start---', str(datetime.datetime.now()))

# %%

# data reads
df_returns_train = pd.read_csv('data/returns_train.csv')
df_returns_test = pd.read_csv('data/returns_test.csv')
df_returns_train['month_end'] = pd.to_datetime(arg=df_returns_train['month_end']).apply(lambda d: d.date())
df_returns_test['month_end'] = pd.to_datetime(arg=df_returns_test['month_end']).apply(lambda d: d.date())

# %%

def equalise_weights(df: pd.DataFrame):

    '''
        Function to generate the equal weights, i.e. 1/p for each active stock within a month

        Args:
            df: A return data frame. First column is month end and remaining columns are stocks

        Returns:
            A dataframe of the same dimension but with values 1/p on active funds within a month

    '''

    # create df to house weights
    n_length = len(df)
    df_returns = df
    df_weights = df_returns[:n_length].copy()
    df_weights.set_index('month_end', inplace=True)

    # list of stock names
    list_stocks = list(df_returns.columns)
    list_stocks.remove('month_end')

    # assign 1/p
    df_weights[list_stocks] = 1/len(list_stocks)

    return df_weights


# %%

def generate_portfolio(df_train: pd.DataFrame, df_test: pd.DataFrame):

    '''
        Function to generate stocks weight allocation for time t+1 using historic data. Initial weights generated as 1/p for active stock within a month

        Args:
            df_train: The training set of returns. First column is month end and remaining columns are stocks
            df_test: The testing set of returns. First column is month end and remaining columns are stocks

        Returns:
            The returns dataframe and the weights
    '''

    print('---> training set spans', df_train['month_end'].min(), df_train['month_end'].max())
    print('---> training set spans', df_test['month_end'].min(), df_test['month_end'].max())

    # initialise data
    n_train = len(df_train)
    df_returns = pd.concat(objs=[df_train, df_test], ignore_index=True)

    df_weights = equalise_weights(df_returns[:n_train]) # df to store weights and create initial

    # list of stock names
    list_stocks = list(df_returns.columns)
    list_stocks.remove('month_end')

    # <<--------------------- YOUR CODE GOES BELOW THIS LINE --------------------->>

    # This is your playground. Delete/modify any of the code here and replace with 
    # your methodology. Below we provide a simple, naive estimation to illustrate 
    # how we think you should go about structuring your submission and your comments:

    # We use a static Inverse Volatility Weighting (https://en.wikipedia.org/wiki/Inverse-variance_weighting) 
    # strategy to generate portfolio weights.
    # Use the latest available data at that point in time
    def totalR(w, returns):        
        tr = 0
        for i in range(54):
            tr += (w[i]*returns[i])
        return tr + 1

    ''' Monte Carlo Approach - works 70% of the time unfortunately
    def optTr(r):
        yM = 0
        xM = []
        for t in range(100000000000):
            x1 = np.random.uniform(0, 0.1, 53)
            max_value = 1 - np.sum(x1)
            x1 = np.append(x1, np.random.uniform(0, max_value)).tolist()

            y = totalR(x1, r)
            if y>yM:
                yM = y
                xM = x1
            return xM, yM
    '''
   # The following code uses a predetermined array of weights which gets reordered depending on which order returns the highest total returns
   # Once the best order of weights is found for the past data, the weights are used for the plot
    def optTr(r):
        yM = 0
        x1 = [0.08919999999999961,0.1,0.08,0.09,0.06,0.06,0.03,0,0.02,0.02,0.02,0.005, 0.005,0.0001,0.0001,0.01,0.01,0.01,0.01,0.01,0.02,0.02,0.02,0.02,0.005, 0.005,0.0001,0.0001,0.01,0.01,0.01,0.01,0.01,0.02,0.02,0.02,0.02,0.005, 0.005,0.0001,0.0001,0.01,0.01,0.01,0.01,0.01,0.02,0.02,0.02,0.02,0.005, 0.005,0.0001,0.0001]
        xM = []
        for t in range(54):
            xtemp = [0.08919999999999961,0.09,0.09,0.09,0.06,0.06,0.03,0,0.02,0.02,0.02,0.005, 0.005,0.0001,0.0001,0.01,0.01,0.01,0.01,0.01,0.02,0,0,0.06,0.005, 0.005,0.0001,0.0001,0.01,0.01,0.01,0.01,0.01,0.02,0.02,0.02,0.02,0.005, 0.005,0.0001,0.0001,0.01,0.01,0.01,0.01,0.01,0.02,0.02,0.02,0.02,0.005, 0.005,0.0001,0.0001]
            for xx in range(len(xM)):
                x1[xx] = xtemp.pop(t)
            
            y = totalR(x1, r)
            if y>yM:
                yM = y
                xM = x1
            return xM, yM

    for i in range(len(df_test)):

        # latest data at this point
        df_latest = df_returns[(df_returns['month_end'] < df_test.loc[i, 'month_end'])]
        df_latest_arr = df_latest.to_numpy().tolist()

        trsX = []
        trsY = []
        for r in df_latest_arr:
            found = optTr(r[1:])
            trsX.append(found[0])
            trsY.append(found[1])

        df_w = pd.DataFrame()
        df_w['weight'] = trsX[i]

        
        # add to all weights
        df_this = pd.DataFrame(data=[[df_test.loc[i, 'month_end']] + df_w['weight'].to_list()], columns=df_latest.columns)
        df_weights = pd.concat(objs=[df_weights, df_this], ignore_index=True)
    
    # <<--------------------- YOUR CODE GOES ABOVE THIS LINE --------------------->>
    
    # 10% limit check
    if len(np.array(df_weights[list_stocks])[np.array(df_weights[list_stocks]) > 0.101]):

        raise Exception(r'---> 10% limit exceeded')

    return df_returns, df_weights


# %%


def plot_total_return(df_returns: pd.DataFrame, df_weights_index: pd.DataFrame, df_weights_portfolio: pd.DataFrame):

    '''
        Function to generate the two total return indices.

        Args:
            df_returns: Ascending date ordered combined training and test returns data.
            df_weights_index: Index weights. Equally weighted
            df_weights_index: Portfolio weights. Your portfolio should use equally weighted for the training date range. If blank will be ignored

        Returns:
            A plot of the two total return indices and the total return indices as a dataframe
    '''

    # list of stock names
    list_stocks = list(df_returns.columns)
    list_stocks.remove('month_end')

    # replace nans with 0 in return array
    ar_returns = np.array(df_returns[list_stocks])
    np.nan_to_num(x=ar_returns, copy=False, nan=0)

    # calc index
    ar_rtn_index = np.array(df_weights_index[list_stocks])*ar_returns
    ar_rtn_port = np.array(df_weights_portfolio[list_stocks])*ar_returns

    v_rtn_index = np.sum(ar_rtn_index, axis=1)
    v_rtn_port = np.sum(ar_rtn_port, axis=1)

    # add return series to dataframe
    df_rtn = pd.DataFrame(data=df_returns['month_end'], columns=['month_end'])
    df_rtn['index'] = v_rtn_index
    df_rtn['portfolio'] = v_rtn_port
    df_rtn

    # create total return
    base_price = 100
    df_rtn.sort_values(by = 'month_end', inplace = True)
    df_rtn['index_tr'] = ((1 + df_rtn['index']).cumprod()) * base_price
    df_rtn['portfolio_tr'] = ((1 + df_rtn['portfolio']).cumprod()) * base_price
    df_rtn

    df_rtn_long = df_rtn[['month_end', 'index_tr', 'portfolio_tr']].melt(id_vars='month_end', var_name='series', value_name='Total Return')

    # plot
    fig1 = px.line(data_frame=df_rtn_long, x='month_end', y='Total Return', color='series')

    return fig1, df_rtn

# %%

# running solution
df_returns = pd.concat(objs=[df_returns_train, df_returns_test], ignore_index=True)
df_weights_index = equalise_weights(df_returns)
df_returns, df_weights_portfolio = generate_portfolio(df_returns_train, df_returns_test)
fig1, df_rtn = plot_total_return(df_returns, df_weights_index=df_weights_index, df_weights_portfolio=df_weights_portfolio)
fig1

# %%
