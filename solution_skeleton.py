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

    #EXPLANATION
    #https://maesela.notion.site/Prescient-48a940d3bff4497db8512468dbe7bc18?pvs=4

    # This is your playground. Delete/modify any of the code here and replace with 
    # your methodology. Below we provide a simple, naive estimation to illustrate 
    # how we think you should go about structuring your submission and your comments:

    # We use a static Inverse Volatility Weighting (https://en.wikipedia.org/wiki/Inverse-variance_weighting) 
    # strategy to generate portfolio weights.
    # Use the latest available data at that point in time

    # Define the column names
    columns = list_stocks

    # Create the empty DataFrame with specified columns
    df_stock_accelerations = {key: [] for key in list_stocks}
    count_a = 0
    for stock_iterator in range(len(list_stocks)):
        
        # stock k's returns
        # k = stock_iterator
        rk = df_returns[list_stocks[stock_iterator]]

        # ema = exponential moving average return
        # alpha = smoothing factor
        def ema(returns_k, previous_ema):
            alpha = 0.2
            return (alpha * returns_k) + ( (1-alpha) * previous_ema ) 
 
        ema_last_month = 0

        for date_iterator in range(2, len(df_returns)):
            # stock k's returns in month t-1 
            # t = date_iterator

            rkt = rk[date_iterator-1]

            # change in exponential moving average
            delta_w = ema (rkt, ema_last_month) -  ema_last_month

            # acceleration at month t
            beta = 15 # must be > 0 and a whole number. Beta is the time period
            a_t = delta_w / beta   
            count_a += 1
            print(count_a)         

            # append the acceleration
            df_stock_accelerations[list_stocks[stock_iterator]].append(a_t)
        
    
    # print(df_stock_accelerations)
    num_accs = len(df_stock_accelerations[list_stocks[0]])
    for row in range(num_accs):
        accelerations = {stock: df_stock_accelerations[stock][row] if df_stock_accelerations[stock] else None for stock in list_stocks}

        raw_weights = {key: 0 for key in list_stocks}
        
        max_acceleration_key = max(accelerations, key=accelerations.get)

        for key, single_acc in accelerations.items():
            if single_acc < 0:
                raw_weights[key] = 0
            else:
                raw_weight_k = 0.1 if (accelerations[key] / accelerations[max_acceleration_key] ) > 0.1 else (accelerations[key] / accelerations[max_acceleration_key] )
                raw_weights[key] = raw_weight_k


        sum_raw_weights = sum(raw_weights.values())
        print("sum of raw weights", raw_weights)
        weights = {key: 0 for key in list_stocks}
        for key, value in raw_weights.items():
            weights[key] = value / sum_raw_weights



        print(weights)
        df_weights = pd.DataFrame(columns=list_stocks)

        # Convert the dictionary to a DataFrame
        new_row_df = pd.DataFrame([weights])

        # Add the new row using the `loc` indexer
        df_weights = pd.concat([df_weights, new_row_df], ignore_index=True)

        # df_weights = df_weights.append(weights, ignore_index=True)

         # add to all weights
        #df_this = pd.DataFrame(data=[[df_test.loc[i, 'month_end']] + df_w['weight'].to_list()], columns=df_latest.columns)
        #df_weights = pd.concat(objs=[df_weights, df_this], ignore_index=True)
    
    # # delete this loop
    # for i in range(len(df_test)):

    #     # latest data at this point
    #     df_latest = df_returns[(df_returns['month_end'] < df_test.loc[i, 'month_end'])]
                
    #     # vol calc
    #     df_w = pd.DataFrame()
    #     df_w['vol'] = df_latest.std(numeric_only=True)          # calculate stock volatility
    #     df_w['inv_vol'] = 1/df_w['vol']                         # calculate the inverse volatility
    #     df_w['tot_inv_vol'] = df_w['inv_vol'].sum()             # calculate the total inverse volatility
    #     df_w['weight'] = df_w['inv_vol']/df_w['tot_inv_vol']    # calculate weight based on inverse volatility
    #     df_w.reset_index(inplace=True, names='name')

    #     # add to all weights
    #     df_this = pd.DataFrame(data=[[df_test.loc[i, 'month_end']] + df_w['weight'].to_list()], columns=df_latest.columns)
    #     df_weights = pd.concat(objs=[df_weights, df_this], ignore_index=True)
    
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
