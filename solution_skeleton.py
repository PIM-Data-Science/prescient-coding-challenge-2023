# %%
import numpy as np
import pandas as pd
import datetime
import plotly.express as px
from scipy.optimize import minimize


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
# Define the objective function to maximize

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
    opt=[]
    ret=[]

    for i in range(len(df_test)):

        # latest data at this point
        df_latest = df_returns[(df_returns['month_end'] < df_test.loc[i, 'month_end'])]
        #print(df_latest)
        stonk = df_latest.drop(df_latest.columns[0], axis=1)
        stonk=stonk.loc[i]
        # vol calc
        df_w = pd.DataFrame()
        df_w['vol'] = df_latest.std(numeric_only=True)          # calculate stock volatility
        df_w['inv_vol'] = 1/df_w['vol']                         # calculate the inverse volatility
        df_w['tot_inv_vol'] = df_w['inv_vol'].sum()
        df_w['weight'] = 1/54*(np.ones(54)) #df_w['inv_vol']/df_w['tot_inv_vol'] # calculate the total inverse volatilitydf_w['weight'] = df_w['inv_vol']/df_w['tot_inv_vol']
        df_w.reset_index(inplace=True, names='name')

        #Choose which weights to use
        if i ==0:
            weights = np.array(df_w['weight'])
        else:
            weights=upweight

        initial_weight=weights.copy()

        #Functions for optimization
        def objective_function(weights, stonk):
            Returns = np.dot(weights, stonk)
            return -Returns  # Minimize the negative of the Returns (maximize the Returns)

        def constraint_sum_to_1(weights):
            return sum(weights) - 1  # Constraint: sum of weights must be 1

        # Additional constraint: weights cannot be zero (must be greater than or equal to 0.0001)
        def constraint_nonzero_weights(weights):
            return weights  # Constraint: weights >= 0.0001

        # Additional constraint: each weight must be less than 0.1
        def constraint_nonnegative_weights(weights):
            return 0.1 - weights  # Constraint: weights <= 0.1 for all weights

        # Combine all constraint functions and types into a list of dictionaries
        constraints = [
            {'type': 'eq', 'fun': constraint_sum_to_1},   # sum of weights must be 1
            {'type': 'ineq', 'fun': constraint_nonzero_weights},
            {'type': 'ineq', 'fun': constraint_nonnegative_weights},  # weights <= 0.1 for all weights
        ]
        
        # Initial weights to optimise
        initial_guess = weights

        # Minimize the negative of the objective function to maximize the original objective
        result = minimize(objective_function, initial_guess, args=(stonk,), constraints=constraints )

        opt_weight=result.x

        Return= np.dot(opt_weight, stonk)
        OldReturn=np.dot(initial_weight, stonk)

        #Check if  the new weights are better. If not continue using old weights
        if Return> OldReturn:
            upweight = opt_weight
        else:
            upweight=initial_weight


        df_w['weight1'] = (opt_weight)
        print(df_w)

        # add to all weights
        df_this = pd.DataFrame(data=[[df_test.loc[i, 'month_end']] + df_w['weight1'].to_list()], columns=df_latest.columns)
        df_weights = pd.concat(objs=[df_weights, df_this], ignore_index=True)


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
fig1.show()
