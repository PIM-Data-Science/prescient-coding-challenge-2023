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

from sklearn.linear_model import LinearRegression

def predict_returns(df_train, df_test):
    model = LinearRegression()
    features = df_train.drop(columns=['month_end'])
    target = df_train.iloc[:, -1]

    model.fit(features, target)
    predicted_returns = model.predict(df_test.drop(columns=['month_end']))

    return predicted_returns

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

    # Combine training and testing data for easier calculations
    df_returns = pd.concat([df_train, df_test], ignore_index=True)

    df_weights = equalise_weights(df_train)  # df to store weights and create initial

    # list of stock names
    list_stocks = df_train.columns.drop('month_end')

    # Loop through each time step in df_test
    for i in range(len(df_test)):
        # Latest data at this point
        df_latest = df_returns[df_returns['month_end'] < df_test.loc[i, 'month_end']]

        # Use machine learning to predict future returns
        predicted_returns = predict_returns(df_latest, df_test.iloc[[i]])

        # vol calc
        df_w = pd.DataFrame()
        df_w['vol'] = df_latest.std(numeric_only=True)          # calculate stock volatility
        df_w['inv_vol'] = 1 / df_w['vol']                        # calculate the inverse volatility
        df_w['tot_inv_vol'] = df_w['inv_vol'].sum()              # calculate the total inverse volatility

        # Apply your custom weighting algorithm here based on inverse volatility and predicted returns
        # Assuming the weights cannot exceed 10% and setting zero weight for negative returns.
        df_w['weight'] = np.where(
            (predicted_returns > 0) & (predicted_returns == predicted_returns),  # Filtering non-NaN positive returns
            (df_w['inv_vol'] / df_w['tot_inv_vol']) * 0.10,
            0.0
        )

        # Increase prioritization weighting on higher return stocks by applying power transformation.
        scaling_factor = 1.5  # Adjust this scaling factor as needed.
        df_w['weight'] = df_w['weight'] ** scaling_factor

        # Normalize the weights to sum up to 1 after applying the scaling factor.
        df_w['weight'] /= df_w['weight'].sum()

        # Reset index to align with the list_stocks
        df_w.reset_index(inplace=True, drop=True)

        # Create a DataFrame to store the results.
        df_this = pd.DataFrame(data=[[df_test.loc[i, 'month_end']] + df_w['weight'].to_list()],
                               columns=df_latest.columns)
        df_weights = pd.concat([df_weights, df_this], ignore_index=True)

    # 10% limit check
    if (df_weights[list_stocks] > 0.101).any().any():
        raise Exception('---> 10% limit exceeded')

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
