# %%
import numpy as np
import pandas as pd
import datetime
import plotly.express as px
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import LSTMModel
from torch.autograd import Variable
import torch

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
    time_step =100
    scaler = MinMaxScaler()
    pred_df = pd.DataFrame(index=df_returns_train.index[time_step+1], columns=list_stocks)
    
    for ticker in list_stocks:
        # normalizing dataset
        norm_act_adj_close = scaler.fit_transform(np.array(df_returns_train[ticker]).reshape(-1, 1))

        train_set = norm_act_adj_close

        X_train, y_train = create_xy(train_set, time_step)
        X_val, y_val = create_xy(norm_act_adj_close, time_step)

        # converting datasets to tensors
        X_train_tensors = Variable(torch.Tensor(X_train))
        y_train_tensors = Variable(torch.Tensor(y_train))

        X_val_tensors = Variable(torch.Tensor(X_val))
        y_true = scaler.inverse_transform(y_val) # inverse transformation
        
        # fitting model
        print('----------')
        print(f'{ticker} fitting...')
        print(f'{len(list_stocks) - (list_stocks.index(ticker) + 1)} left')
        model = fit_model(ticker=ticker, X_train=X_train_tensors, y_train=y_train_tensors)

        # predicting response
        y_hat = model(X_val_tensors) # forward pass
        y_hat = y_hat.data.detach().cpu().numpy() # numpy conversion
        y_hat = scaler.inverse_transform(y_hat) # inverse transformation
        print('Done!')
        
        # pred_adj_close_df[ticker] = y_hat
        
        # model_perf_df[ticker] = [MAE(y_true, y_hat), MSE(y_true, y_hat), RMSE(y_true, y_hat), 
        #                         MAPE(y_true, y_hat), MPE(y_true, y_hat)]

        rand_ticker = np.random.choice(list_stocks)
        my_fig = px.plot(pred_df.index, pred_df[rand_ticker][time_step + 1:], label='Actual Price')


    # This is your playground. Delete/modify any of the code here and replace with 
    # your methodology. Below we provide a simple, naive estimation to illustrate 
    # how we think you should go about structuring your submission and your comments:

    # We use a static Inverse Volatility Weighting (https://en.wikipedia.org/wiki/Inverse-variance_weighting) 
    # strategy to generate portfolio weights.
    # Use the latest available data at that point in time
    my_fig.show()

    for i in range(len(df_test)):

        # latest data at this point
        df_latest = df_returns[(df_returns['month_end'] < df_test.loc[i, 'month_end'])]
                
        # vol calc
        df_w = pd.DataFrame()
        df_w['vol'] = df_latest.std(numeric_only=True)          # calculate stock volatility
        df_w['inv_vol'] = 1/df_w['vol']                         # calculate the inverse volatility
        df_w['tot_inv_vol'] = df_w['inv_vol'].sum()             # calculate the total inverse volatility
        df_w['weight'] = df_w['inv_vol']/df_w['tot_inv_vol']    # calculate weight based on inverse volatility
        df_w.reset_index(inplace=True, names='name')

        # add to all weights
        df_this = pd.DataFrame(data=[[df_test.loc[i, 'month_end']] + df_w['weight'].to_list()], columns=df_latest.columns)
        df_weights = pd.concat(objs=[df_weights, df_this], ignore_index=True)
    
    # <<--------------------- YOUR CODE GOES ABOVE THIS LINE --------------------->>
    
    # 10% limit check
    if len(np.array(df_weights[list_stocks])[np.array(df_weights[list_stocks]) > 0.101]):

        raise Exception(r'---> 10% limit exceeded')

    return df_returns, df_weights

# function for fitting model
def fit_model(ticker, X_train, y_train):
    model, optimizer, loss_function = build_model()
    num_epochs = 10000
    
    loss_list = []

    for epoch in range(num_epochs + 1):
        outputs = model.forward(X_train) # forward pass
        optimizer.zero_grad() # calculating gradient, manually setting to 0
        loss = loss_function(outputs, y_train) # obtaining loss
        loss.backward() # calculating loss of loss function
        optimizer.step() # improving from loss, i.e. backprop
        
        if epoch % 1000 == 0:
            loss_list.append(loss.item())
    

    # saving model state
    torch.save(model.state_dict(), f'model-states/{ticker}-model-state.pth')
    
    return model

def MSE(y_true, y_hat):
    return np.mean(np.square(y_true - y_hat))

def RMSE(y_true, y_hat):
    return np.sqrt(MSE(y_true, y_hat))

def create_xy(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:i + time_step])
        y.append(data[i + time_step])
    return np.array(X), np.array(y)

def build_model(input_size=1, hidden_size=1, num_layers=1, num_classes=1, learning_rate=0.001):
    # learning rate controls how much to change model in response to estm error 
    # each time model weights are updated
    
    model = LSTMModel(input_size, hidden_size, num_layers, num_classes)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # adam optimizer
    # algorithms/methods to change attributes of neural network such as weights and learning rate to reduce losses

    loss_function = torch.nn.MSELoss() # mean-squared error of regression
    # loss function measures how bad model performs: high loss -> low accuracy

    # loading model state
    # model = LSTM_model(*args, **kwargs)
    # model.load_state_dict(torch.load(PATH))
    # model.eval()

    return model, optimizer, loss_function


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
    fig1.show()
    
    return fig1, df_rtn

# %%

# running solution
df_returns = pd.concat(objs=[df_returns_train, df_returns_test], ignore_index=True)
df_weights_index = equalise_weights(df_returns)
df_returns, df_weights_portfolio = generate_portfolio(df_returns_train, df_returns_test)
fig1, df_rtn = plot_total_return(df_returns, df_weights_index=df_weights_index, df_weights_portfolio=df_weights_portfolio)
fig1.show()
