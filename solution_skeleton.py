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

df_returns_train
df_returns_test

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
    
    
    import tensorflow as tf
    from tensorflow import keras
    from tensorboard.plugins.hparams import api as hp
    
    
    #remove dates column 
    df_test_Clean = df_returns_test.iloc[:, 1:56]
    df_train_Clean = df_returns_train.iloc[:, 1:56]
    df_test_Clean
    df_train_Clean

    
    # Combine train and test data for preprocessing
    df_returns = pd.concat([df_train_Clean, df_test_Clean])
    df_returns


    # Normalize the returns data
    returns_array = df_returns.values
    returns_mean = np.mean(returns_array, axis=0)
    returns_std = np.std(returns_array, axis=0)
    returns_array_normalized = (returns_array - returns_mean) / returns_std
    
    # Split the data back into train and test sets
    df_returns_train_normalized = returns_array_normalized[:len(df_returns_train)]
    df_returns_test_normalized = returns_array_normalized[len(df_returns_train):]

   


    HP_L2 = hp.HParam('l2_regulariser', hp.RealInterval(0.01,0.02))
    HP_ACTIVATIONS = hp.HParam('Activations', hp.Discrete(['sigmoid','elu','tanh','softmax','softplus','relu']))
    HP_Layer_1_Nodes = hp.HParam('Layer_1_Nodes', hp.Discrete([20, 40, 60, 80, 100])) 
    HP_Layer_2_Nodes = hp.HParam('Layer_2_Nodes', hp.Discrete([20, 40, 60, 80, 100]))
    
    
    with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
      hp.hparams_config(
        hparams=[HP_Layer_1_Nodes, HP_Layer_2_Nodes, HP_L2 ,HP_ACTIVATIONS],
       
      )
        
     
   
    # Define the neural network model
    def create_model(hparams):
        model = tf.keras.Sequential([
                tf.keras.layers.Dense(HP_Layer_1_Nodes, activation=HP_ACTIVATIONS),
                tf.keras.layers.Dense(HP_Layer_2_Nodes, activation=HP_ACTIVATIONS ),
                tf.keras.layers.Dense(len(df_returns.columns), activation='softmax')
                ])
      
        return model 
    
    
    def train_val_model(hparams):
        model = create_model(hparams)
        model.compile(optimizer="adam", loss='mean_squared_error', metrics=['mean_absolute_error'])
        model.fit(df_returns_train_normalized, df_returns_test_normalized, epochs=100, batch_size=8)
        loss_and_metrics = model.evaluate(inputs_val, outputs_val)
        return loss_and_metrics[1], loss_and_metrics[2]  # Return MAE and MSE
    
   
    # Grid search over hyperparameter combinations
    session_num = 0
    best_combined_metric = float('inf')
    best_hparams = {}
    for Layer_1_Nodes in HP_Layer_1_Nodes.domain.values:
        for Layer_2_Nodes in HP_Layer_2_Nodes.domain.values:
            for l2_regulariser in np.linspace(HP_L2.domain.min_value, HP_L2.domain.max_value, num=5):
                for Activations in HP_ACTIVATIONS.domain.values:
                    hparams = {
                        HP_Layer_1_Nodes: Layer_1_Nodes,
                        HP_Layer_2_Nodes: Layer_2_Nodes,
                        HP_L2: l2_regulariser,
                        HP_ACTIVATIONS: Activations, 
                     }
    
   
    
   # Compile the model
   
    model.compile(optimizer='adam', loss="mean_absolute_error", metrics=['accuracy'])
    
    # Prepare training data and labels
    Past_train = df_returns_train_normalized[:-1]  # training data all the way up to last month 
    Future_train = df_returns_train_normalized[1:]   # Predict weights for future month 
    
    # Convert y_train to one-hot encoding
    Future_train_onehot = np.zeros_like(Future_train)
    Future_train_onehot[np.arange(len(Future_train)), np.argmax(Future_train, axis=1)] = 1
    
    # Train the model
    model.fit(Past_train, Future_train_onehot, epochs=100, batch_size=8)
    
    # Use the model to predict portfolio weights for the test set
    Past_test = df_returns_test_normalized[:-1]  # Use all but the last month as test data
    Future_test_predicted = model.predict(Past_test)
    
    # Normalize the predicted weights so that they sum to 1 for each month
    Future_test_normalized = Future_test_predicted / np.sum(Future_test_predicted, axis=1, keepdims=True)
    
    # Convert the normalized weights back to original scale
    df_weights = Future_test_normalized * returns_std + returns_mean
    
    # Optionally, you can enforce constraints on the weights (e.g., limit to 10%)
    df_weights[df_weights > 0.1] = 0.1
    
    # The resulting y_test_weights is the generated portfolio weights for each month in the test set 




   
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
