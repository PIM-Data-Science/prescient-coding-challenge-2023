import numpy as np
import pandas as pd
import datetime
import plotly.express as px
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print("---Python script Start---", str(datetime.datetime.now()))

df = pd.read_csv(
    "C:\\Users\\terre\OneDrive - University of Cape Town\\Terrence\\Code\\prescient-coding-challenge-2023\\data\\returns_train.csv"
)

# data reads
df_returns_train = pd.read_csv("data/returns_train.csv")
df_returns_test = pd.read_csv("data/returns_test.csv")
df_returns_train["month_end"] = pd.to_datetime(arg=df_returns_train["month_end"]).apply(
    lambda d: d.date()
)
df_returns_test["month_end"] = pd.to_datetime(arg=df_returns_test["month_end"]).apply(
    lambda d: d.date()
)

# Ignoring the month_end from training set

# Get column names as a list but ignore the first column
header_list = df.columns[1:].tolist()
print(header_list)


def process_dataframe(dataframe):
    # Create a new dataframe without the 'month_end' column
    new_dataframe = dataframe.drop(["month_end"], axis=1)
    return new_dataframe


clean_df = process_dataframe(df_returns_train)


def equalise_weights(df: pd.DataFrame):
    """
    Function to generate the equal weights, i.e. 1/p for each active stock within a month

    Args:
        df: A return data frame. First column is month end and remaining columns are stocks

    Returns:
        A dataframe of the same dimension but with values 1/p on active funds within a month

    """

    # create df to house weights
    n_length = len(df)
    df_returns = df
    df_weights = df_returns[:n_length].copy()
    df_weights.set_index("month_end", inplace=True)

    # list of stock names
    list_stocks = list(df_returns.columns)
    list_stocks.remove("month_end")

    # assign 1/p
    df_weights[list_stocks] = 1 / len(list_stocks)

    return df_weights


# Normalization
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(clean_df.values)

# Create a data structure with 60 time-steps and 1 output for each stock
lookback = 60

X, y = [], []
for i in range(lookback, len(scaled_data)):
    X.append(scaled_data[i - lookback : i])
    y.append(scaled_data[i])
X, y = np.array(X), np.array(y)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the LSTM model
model = Sequential()

model.add(
    LSTM(
        units=50,
        return_sequences=True,
        input_shape=(X_train.shape[1], X_train.shape[2]),
    )
)
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(
    Dense(units=clean_df.shape[1])
)  # number of units in the output layer should be equal to the number of stocks

# Compile and train the model
model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X_train, y_train, epochs=50, batch_size=20, validation_data=(X_val, y_val))


def generate_portfolio(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """
    Function to generate stocks weight allocation for time t+1 using historic data. Initial weights generated as 1/p for active stock within a month

    Args:
        df_train: The training set of returns. First column is month end and remaining columns are stocks
        df_test: The testing set of returns. First column is month end and remaining columns are stocks

    Returns:
        The returns dataframe and the weights
    """

    print(
        "---> training set spans",
        df_train["month_end"].min(),
        df_train["month_end"].max(),
    )
    print(
        "---> training set spans",
        df_test["month_end"].min(),
        df_test["month_end"].max(),
    )

    # initialise data
    n_train = len(df_train)
    df_returns = pd.concat(objs=[df_train, df_test], ignore_index=True)

    df_weights = equalise_weights(
        df_returns[:n_train]
    )  # df to store weights and create initial

    # list of stock names
    list_stocks = list(df_returns.columns)
    list_stocks.remove("month_end")

    # Only positive predicted stock are considered
    # Higher volitility have higher weighting
    # Higher prediction through my machine learning algorithm (LSTM) will have higher weighting
    for i in range(len(df_test)):
        df_latest = df_returns[(df_returns["month_end"] < df_test.loc[i, "month_end"])]

        if len(df_latest) >= lookback:
            latest_returns = df_latest[-lookback:][list_stocks].values
        else:
            continue

        # reshape the latest returns to be suitable for the LSTM model
        latest_returns = np.reshape(latest_returns, (1, lookback, len(list_stocks)))

        # predict the future returns using the LSTM model
        predicted_returns = model.predict(latest_returns)

        # Filter out the stocks with positive predicted returns
        positive_returns_indices = np.where(predicted_returns[0] > 0)[0]
        positive_returns = predicted_returns[0][positive_returns_indices]
        positive_stock_names = np.array(list_stocks)[positive_returns_indices]

        # Apply exponential weighting to the positive returns
        exp_returns = np.exp(positive_returns)

        # Calculate the recent volatility for each stock
        volatilities = np.std(latest_returns[0], axis=0)

        # Scale the weights of each stock correspondto its recent volatility
        volatility_scaled_weights = exp_returns * volatilities[positive_returns_indices]

        # Normalize the weights to sum to 1
        weights = volatility_scaled_weights / np.sum(volatility_scaled_weights)

        # Create a new weight dataframe
        weights_dict = dict(zip(positive_stock_names, weights))
        weights_dict["month_end"] = df_test.loc[i, "month_end"]
        df_this = pd.DataFrame(data=[weights_dict], columns=df_weights.columns)
        df_this = df_this.fillna(0)  # fill NaNs for stocks that have been excluded

        # Append the weights to the df_weights dataframe
        df_weights = pd.concat(objs=[df_weights, df_this], ignore_index=True)

    # 10% limit check
    if len(
        np.array(df_weights[list_stocks])[np.array(df_weights[list_stocks]) > 0.101]
    ):
        raise Exception(r"---> 10% limit exceeded")

    return df_returns, df_weights


def plot_total_return(
    df_returns: pd.DataFrame,
    df_weights_index: pd.DataFrame,
    df_weights_portfolio: pd.DataFrame,
):
    """
    Function to generate the two total return indices.

    Args:
        df_returns: Ascending date ordered combined training and test returns data.
        df_weights_index: Index weights. Equally weighted
        df_weights_index: Portfolio weights. Your portfolio should use equally weighted for the training date range. If blank will be ignored

    Returns:
        A plot of the two total return indices and the total return indices as a dataframe
    """

    # list of stock names
    list_stocks = list(df_returns.columns)
    list_stocks.remove("month_end")

    # replace nans with 0 in return array
    ar_returns = np.array(df_returns[list_stocks])
    np.nan_to_num(x=ar_returns, copy=False, nan=0)

    # calc index
    ar_rtn_index = np.array(df_weights_index[list_stocks]) * ar_returns
    ar_rtn_port = np.array(df_weights_portfolio[list_stocks]) * ar_returns

    v_rtn_index = np.sum(ar_rtn_index, axis=1)
    v_rtn_port = np.sum(ar_rtn_port, axis=1)

    # add return series to dataframe
    df_rtn = pd.DataFrame(data=df_returns["month_end"], columns=["month_end"])
    df_rtn["index"] = v_rtn_index
    df_rtn["portfolio"] = v_rtn_port
    df_rtn

    # create total return
    base_price = 100
    df_rtn.sort_values(by="month_end", inplace=True)
    df_rtn["index_tr"] = ((1 + df_rtn["index"]).cumprod()) * base_price
    df_rtn["portfolio_tr"] = ((1 + df_rtn["portfolio"]).cumprod()) * base_price
    df_rtn

    df_rtn_long = df_rtn[["month_end", "index_tr", "portfolio_tr"]].melt(
        id_vars="month_end", var_name="series", value_name="Total Return"
    )

    # plot
    fig1 = px.line(
        data_frame=df_rtn_long, x="month_end", y="Total Return", color="series"
    )

    return fig1, df_rtn


# running solution
df_returns = pd.concat(objs=[df_returns_train, df_returns_test], ignore_index=True)
df_weights_index = equalise_weights(df_returns)
df_returns, df_weights_portfolio = generate_portfolio(df_returns_train, df_returns_test)
fig1, df_rtn = plot_total_return(
    df_returns,
    df_weights_index=df_weights_index,
    df_weights_portfolio=df_weights_portfolio,
)
fig1
