import numpy as np
import pandas as pd

def append_rolling_outlier_signal(df, identifier, asset_value_column='Close', window_size=30, multiplier=1.5):
    """
    Appends a binary signal column to the DataFrame to indicate outliers within a rolling window 
    for a specific identifier based on the asset value column.

    This function calculates rolling quartiles and IQR within the specified window size. It then 
    determines outliers as those values that lie below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR within
    the rolling window. The outliers are marked in a new binary column specific to the identifier, 
    window size, and multiplier used.

    :param df: DataFrame - The data containing the asset values.
    :param identifier: str - A unique identifier for the data subset within the DataFrame, used in naming the outlier column.
    :param asset_value_column: str - The name of the column with asset values.
    :param window_size: int - The size of the rolling window to calculate the IQR and detect outliers.
    :param multiplier: float - The multiplier for the IQR to set thresholds for outliers within the rolling window.
                               Typically, 1.5 is used, but can be adjusted if needed.
    :return: DataFrame - The original data with an added binary column indicating rolling window outliers, 
                         named according to the identifier, window size, and multiplier.
    """
    # Calculate rolling quartiles and IQR
    rolling_Q1 = df[f'{asset_value_column}_{identifier}'].rolling(window=window_size).quantile(0.25)
    rolling_Q3 = df[f'{asset_value_column}_{identifier}'].rolling(window=window_size).quantile(0.75)
    rolling_IQR = rolling_Q3 - rolling_Q1

    # Determine the rolling lower and upper bounds for outliers
    rolling_lower_bound = rolling_Q1 - (multiplier * rolling_IQR)
    rolling_upper_bound = rolling_Q3 + (multiplier * rolling_IQR)

    # Construct the name for the outlier column incorporating the identifier, window size, and multiplier
    outlier_col_name = f'RollingOutlier_{identifier}_{asset_value_column}_Window{window_size}_Multiplier{multiplier}'
    
    # Create the binary signal for rolling window outliers
    df[outlier_col_name] = ((df[f'{asset_value_column}_{identifier}'] < rolling_lower_bound) | (df[f'{asset_value_column}_{identifier}'] > rolling_upper_bound)).astype(int)
    
    return df

def calculate_SMA(df, identifier, windows=[3, 5, 8, 10, 11, 15, 20, 50, 100, 200], columnName='Close'):
    for window in windows:
        df[f'SMA_{columnName}_{window}_{identifier}'] = df[f'{columnName}_{identifier}'].rolling(window=window).mean()
        
        # Replace infinite values with 0
        df[f'SMA_{columnName}_{window}_{identifier}'].replace([np.inf, -np.inf], 0, inplace=True)
        
        # Replace NaN values with 0
        df[f'SMA_{columnName}_{window}_{identifier}'].fillna(0, inplace=True)

    return df

def calculate_EMA(df, identifier, spans=[3, 5, 8, 10, 11, 15, 20, 50, 100, 200], columnName='Close'):
    for span in spans:
        df[f'EMA_{columnName}_{span}_{identifier}'] = df[f'{columnName}_{identifier}'].ewm(span=span, adjust=False).mean()
        
        # Replace infinite values with 0
        df[f'EMA_{columnName}_{span}_{identifier}'].replace([np.inf, -np.inf], 0, inplace=True)
        
        # Replace NaN values with 0
        df[f'EMA_{columnName}_{span}_{identifier}'].fillna(0, inplace=True)

    return df

def calculate_return(df, identifier, periods=[5, 15, 60, 240, 1440], columnName='Close'):
    """
    Calculate returns for cryptocurrency data over multiple periods.

    :param df: DataFrame - The data containing price information.
    :param identifier: str - A unique identifier for the cryptocurrency.
    :param periods: list of int - The list of periods over which to calculate returns. Default is [1].
    :param columnName: str - The name of the column with price data. Default is 'Close'.
    :return: DataFrame - The original data with added columns for returns for each period in periods.
    """
    for period in periods:
        df[f'Return_{columnName}_{period}_{identifier}'] = df[f'{columnName}_{identifier}'].pct_change(periods=period) * 100
        
        # Replace infinite values with 0
        df[f'Return_{columnName}_{period}_{identifier}'].replace([np.inf, -np.inf], 0, inplace=True)
        
        # Replace NaN values with 0
        df[f'Return_{columnName}_{period}_{identifier}'].fillna(0, inplace=True)
        
    return df

def calculate_price_change(df, identifier, price_column='Close'):
    """
    Calculate the amount of price change from the previous period for a specified price column.

    :param df: DataFrame - The data containing the asset prices.
    :param identifier: str - A unique identifier for the data subset within the DataFrame, used in naming the price change column.
    :param price_column: str - The name of the column with asset prices. Default is 'Close'.
    :return: DataFrame - The original data with an added column indicating the price change from the previous period.
    """
    # Calculate the price change from the previous period
    df[f'PriceChange_{identifier}'] = df[f'{price_column}_{identifier}'] - df[f'{price_column}_{identifier}'].shift(1)

    # Replace NaN values with 0 in the first row (since there's no previous data for the first entry)
    df[f'PriceChange_{identifier}'].fillna(0, inplace=True)
    
    return df

def calculate_price_roc(df, identifier, periods=[1, 5, 15, 240, 1440], columnName='Close'):
    """
    Calculate the Price Rate of Change for cryptocurrency data over multiple periods.

    :param df: DataFrame - The data containing price information.
    :param identifier: str - A unique identifier for the cryptocurrency.
    :param periods: list of int - The list of periods over which to calculate Price ROC. Default is [1].
    :param columnName: str - The name of the column with price data. Default is 'Close'.
    :return: DataFrame - The original data with added columns for Price ROC for each period in periods.
    """
    for period in periods:
        df[f'PriceROC_{columnName}_{period}_{identifier}'] = (df[f'{columnName}_{identifier}'] - df[f'{columnName}_{identifier}'].shift(period)) / df[f'{columnName}_{identifier}'].shift(period) * 100
        
        # Replace infinite values with 0
        df[f'PriceROC_{columnName}_{period}_{identifier}'].replace([np.inf, -np.inf], 0, inplace=True)
        
        # Replace NaN values with 0
        df[f'PriceROC_{columnName}_{period}_{identifier}'].fillna(0, inplace=True)
        
    return df

def calculate_volume_change(df, identifier, periods=[1, 5, 15], columnName='Volume'):
    """
    Calculate the Volume Change for cryptocurrency data over multiple periods.

    :param df: DataFrame - The data containing volume information.
    :param identifier: str - A unique identifier for the cryptocurrency.
    :param periods: list of int - The list of periods over which to calculate Volume Change. Default is [1].
    :param columnName: str - The name of the column with volume data. Default is 'Volume'.
    :return: DataFrame - The original data with added columns for Volume Change for each period in periods.
    """
    volume_change_dfs = []  # List to store individual volume change DataFrames
    for period in periods:
        col_name = f'VolumeChange_{columnName}_{period}_{identifier}'
        volume_change = df[f'{columnName}_{identifier}'].diff(period)
        volume_change.replace([np.inf, -np.inf], 0, inplace=True)
        volume_change.fillna(0, inplace=True)
        volume_change_dfs.append(volume_change.to_frame(name=col_name))
    
    # Concatenate all volume change DataFrames at once
    df = pd.concat([df] + volume_change_dfs, axis=1)
    return df

def calculate_volume_roc(df, identifier, ns=[1, 5, 15, 60, 240], columnName='Volume'):
    """
    Calculate the Volume Rate of Change for cryptocurrency data.
    :param df: DataFrame - The data containing volume information.
    :param identifier: str - A unique identifier for the cryptocurrency.
    :param ns: list of int - The list of periods over which to calculate Volume ROC. Default is [10].
    :param columnName: str - The name of the column with volume data. Default is 'Volume'.
    :return: DataFrame - The original data with added columns for Volume ROC for each n in ns.
    """
    roc_dfs = []  # List to hold individual ROC DataFrames
    for n in ns:
        roc_col_name = f'VolumeROC_{columnName}_{n}_{identifier}'
        df_roc = (df[f'{columnName}_{identifier}'] - df[f'{columnName}_{identifier}'].shift(n)) / df[f'{columnName}_{identifier}'].shift(n) * 100
        df_roc.replace([np.inf, -np.inf], 0, inplace=True)
        df_roc.fillna(0, inplace=True)
        roc_dfs.append(df_roc.to_frame(name=roc_col_name))
    
    # Concatenate all ROC DataFrames at once
    df = pd.concat([df] + roc_dfs, axis=1)
    return df

def calculate_historical_volatility(df, identifier, windows=[15, 30, 60, 1440], columnName='Close'):
    """
    Calculate historical volatility for cryptocurrency data using the standard deviation of returns.
    :param df: DataFrame - The data containing price information.
    :param identifier: str - A unique identifier for the cryptocurrency.
    :param windows: list of int - The list of numbers of periods over which to calculate volatility.
    :param columnName: str - The name of the column with price data. Default is 'Close'.
    :return: DataFrame - The original data with added columns for historical volatility for each window size.
    """
    # Calculate log returns
    log_return_col_name = f'LogReturn_{columnName}_{identifier}'
    df[log_return_col_name] = np.log(df[f'{columnName}_{identifier}'] / df[f'{columnName}_{identifier}'].shift(1))

    # Annualizing factor
    annualizing_factor = np.sqrt(365 * 24 * 60)

    # List to store volatility DataFrames
    vol_dfs = []

    # Calculate historical volatility for each window
    for window in windows:
        vol_col_name = f'HistVol_{columnName}_{window}_{identifier}'
        vol_df = df[log_return_col_name].rolling(window=window).std() * annualizing_factor
        vol_df.replace([np.inf, -np.inf], 0, inplace=True)
        vol_df.fillna(0, inplace=True)
        vol_dfs.append(vol_df.to_frame(name=vol_col_name))
    
    # Concatenate all volatility DataFrames at once
    df = pd.concat([df] + vol_dfs, axis=1)
    return df

def calculate_RSI(df, identifier):
    # Calculating price difference
    delta = df[f'Close_{identifier}'].diff(1)
    
    # Separating positive and negative price differences
    up = delta.where(delta > 0, 0)
    down = -delta.where(delta < 0, 0)
    
    # Calculating 4858-minute average gain and average loss
    avg_gain = up.rolling(window=4858).mean()
    avg_loss = down.rolling(window=4858).mean()
    
    # Calculating relative strength
    rs = avg_gain / avg_loss
    
    # Calculating RSI
    df[f'RSI_{identifier}'] = 100 - (100 / (1 + rs))
    
    # Replacing NaN and Inf values with 0
    df[f'RSI_{identifier}'].replace([np.inf, -np.inf], 0, inplace=True)
    
    # Filling NaN values with 0 as required
    df.fillna(0, inplace=True)
    
    return df

def calculate_MACD(df, identifier, short_span=8, long_span=17, signal_span=9):
    df[f'EMA_{short_span}_{identifier}'] = df[f'Close_{identifier}'].ewm(span=short_span, adjust=False).mean()
    df[f'EMA_{long_span}_{identifier}'] = df[f'Close_{identifier}'].ewm(span=long_span, adjust=False).mean()
    df[f'MACD_{identifier}'] = df[f'EMA_{short_span}_{identifier}'] - df[f'EMA_{long_span}_{identifier}']
    df[f'Signal Line_{identifier}'] = df[f'MACD_{identifier}'].ewm(span=signal_span, adjust=False).mean()

def calculate_MACD_Dataset(df, identifier, short_span=8, long_span=17, signal_span=9):
    # Compute EMAs for short and long spans
    df[f'EMA_{short_span}_{identifier}'] = df[f'Close_{identifier}'].ewm(span=short_span, adjust=False).mean()
    df[f'EMA_{long_span}_{identifier}'] = df[f'Close_{identifier}'].ewm(span=long_span, adjust=False).mean()

    # Compute the MACD line as the difference between the two EMAs
    df[f'MACD_{short_span}_{long_span}_{identifier}'] = df[f'EMA_{short_span}_{identifier}'] - df[f'EMA_{long_span}_{identifier}']

    # Compute the Signal line as the EMA of the MACD line
    df[f'MACD_Signal_Line_{short_span}_{long_span}_{signal_span}_{identifier}'] = df[f'MACD_{short_span}_{long_span}_{identifier}'].ewm(span=signal_span, adjust=False).mean()

    # Handle infinite and NaN values
    df[f'MACD_{short_span}_{long_span}_{identifier}'].replace([np.inf, -np.inf], 0, inplace=True)
    df[f'MACD_Signal_Line_{short_span}_{long_span}_{signal_span}_{identifier}'].replace([np.inf, -np.inf], 0, inplace=True)
    
    df[f'MACD_{short_span}_{long_span}_{identifier}'].fillna(0, inplace=True)
    df[f'MACD_Signal_Line_{short_span}_{long_span}_{signal_span}_{identifier}'].fillna(0, inplace=True)

    return df

def calculate_VWAP(df, identifier, sequence_lengths=[3, 5, 10, 15, 20, 30, 60]):
    """
    Calculate the VWAPs for given sequence lengths.

    Parameters:
    - df: DataFrame containing the price and volume data.
    - identifier: Column identifier to distinguish different sets of data in the DataFrame.
    - sequence_lengths: List of sequence lengths for which to calculate the VWAP.

    Returns:
    - DataFrame with the VWAP columns added for each sequence length.
    """
    
    for sequence_length in sequence_lengths:
        # Compute the product of price and volume
        df[f'VP_{sequence_length}_{identifier}'] = df[f'Close_{identifier}'] * df[f'Volume_{identifier}']

        # Compute rolling sums for VP and Volume for the current sequence_length
        rolling_vp = df[f'VP_{sequence_length}_{identifier}'].rolling(window=sequence_length).sum()
        rolling_volume = df[f'Volume_{identifier}'].rolling(window=sequence_length).sum()

        # Calculate rolling VWAP
        df[f'VWAP_{sequence_length}_{identifier}'] = rolling_vp / rolling_volume

        # Handle infinite and NaN values
        df[f'VWAP_{sequence_length}_{identifier}'].replace([np.inf, -np.inf], 0, inplace=True)
        df[f'VWAP_{sequence_length}_{identifier}'].fillna(0, inplace=True)

        # Drop the 'VP_{sequence_length}_{identifier}' column as it's not needed after VWAP calculation
        df.drop(columns=[f'VP_{sequence_length}_{identifier}'], inplace=True)
    
    return df

def calculate_BB(df, identifier, columnName='Close', windows=[20], num_std=2):
    """
    Calculate Bollinger Bands for cryptocurrency data over multiple windows.

    :param df: DataFrame - The data containing price information.
    :param identifier: str - A unique identifier for the cryptocurrency.
    :param columnName: str - The name of the column with price data. Default is 'Close'.
    :param windows: list of int - The list of windows over which to calculate Bollinger Bands. Default is [20].
    :param num_std: int - The number of standard deviations to use for the bands. Default is 2.
    :return: DataFrame - The original data with added columns for Bollinger Bands for each window in windows.
    """
    new_columns = {}
    for window in windows:
        sma = df[f'{columnName}_{identifier}'].rolling(window=window).mean()
        std = df[f'{columnName}_{identifier}'].rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)

        new_columns.update({
            f'SMA_{columnName}_{window}_{identifier}': sma.fillna(0),
            f'STD_{columnName}_{window}_{identifier}': std.fillna(0),
            f'UpperBand_{columnName}_{window}_{identifier}': upper_band.fillna(0),
            f'LowerBand_{columnName}_{window}_{identifier}': lower_band.fillna(0)
        })
        
    df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
    return df

def calculate_AO(df, identifier):
    """
    Calculate the Awesome Oscillator (AO) for cryptocurrency data.

    The AO is calculated as the difference between a short period and a long period 
    moving average of the midpoint (average) of the bars (high + low) / 2. 
    This function adjusts the periods to account for a 24/7 market.

    :param df: DataFrame - The data containing High and Low price information.
    :param identifier: str - A unique identifier for the cryptocurrency.
    :return: DataFrame - The original data with the added column for AO for the identifier.
    """
    # Convert periods from traditional market days to data for 24/7 market
    short_period = 5  # Original 5 bars in typical markets
    long_period = 34  # Original 34 bars in typical markets

    # Adjust periods for 24/7 markets (traditional markets operate for 390 minutes (6.5 hours) a day)
    converted_short_period = (short_period * 390) // 1440
    converted_long_period = (long_period * 390) // 1440
    
    # Calculate the midpoint of the bars
    df[f'midpoint_{identifier}'] = (df[f'High_{identifier}'] + df[f'Low_{identifier}']) / 2
    
    # Calculate the moving averages for the converted periods
    ma_short = df[f'midpoint_{identifier}'].rolling(window=converted_short_period).mean()
    ma_long = df[f'midpoint_{identifier}'].rolling(window=converted_long_period).mean()
    
    # Calculating the Awesome Oscillator value
    df[f'AO_{identifier}'] = ma_short - ma_long
    
    # Replace NaN values with 0
    df.fillna(0, inplace=True)
    
    return df

def calculate_DonchianChannels(df, identifier, periods=[20]):
    """
    Calculate Donchian Channels for the DataFrame based on multiple periods.
    
    Parameters:
    - df: A pandas DataFrame containing high and low price data.
    - identifier: A string to identify the specific columns.
    - periods: A list of integers representing the periods to consider for the Donchian Channels. Default is [20].
    
    Returns:
    - A DataFrame with added Donchian Channel columns.
    """
    
    # Convert periods from traditional market days to 1M data for 24/7 market
    # Considering traditional markets operate for 390 minutes (6.5 hours) a day, adjust periods for 24/7 markets
    converted_periods = [(period * 390) // 1440 for period in periods]

    new_columns = {}
    for converted_period in converted_periods:
        upper_channel = df[f'High_{identifier}'].rolling(window=converted_period).max().fillna(0)
        lower_channel = df[f'Low_{identifier}'].rolling(window=converted_period).min().fillna(0)
        middle_channel = (upper_channel + lower_channel) / 2
        
        new_columns.update({
            f'Donchian_Upper_{converted_period}_{identifier}': upper_channel,
            f'Donchian_Lower_{converted_period}_{identifier}': lower_channel,
            f'Donchian_Middle_{converted_period}_{identifier}': middle_channel
        })

    df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
    return df

def calculate_ADX(df, identifier="", period=7):
    col_prefix = f"{identifier}_" if identifier else ""

    df[f'{col_prefix}Up Move'] = df[f'{col_prefix}High'] - df[f'{col_prefix}High'].shift(1)
    df[f'{col_prefix}Down Move'] = df[f'{col_prefix}Low'].shift(1) - df[f'{col_prefix}Low']
    df[f'{col_prefix}Zero'] = 0

    df[f'{col_prefix}Plus DM'] = np.where((df[f'{col_prefix}Up Move'] > df[f'{col_prefix}Down Move']) & 
                                          (df[f'{col_prefix}Up Move'] > df[f'{col_prefix}Zero']),
                                          df[f'{col_prefix}Up Move'], 0)
    df[f'{col_prefix}Minus DM'] = np.where((df[f'{col_prefix}Up Move'] < df[f'{col_prefix}Down Move']) & 
                                           (df[f'{col_prefix}Down Move'] > df[f'{col_prefix}Zero']),
                                           df[f'{col_prefix}Down Move'], 0)

    df[f'{col_prefix}Plus DM EMA'] = df[f'{col_prefix}Plus DM'].ewm(span=period, adjust=False).mean()
    df[f'{col_prefix}Minus DM EMA'] = df[f'{col_prefix}Minus DM'].ewm(span=period, adjust=False).mean()

    df[f'{col_prefix}True Range'] = np.where((df[f'{col_prefix}High'] - df[f'{col_prefix}Low']) > 
                                             (df[f'{col_prefix}High'] - df[f'{col_prefix}Close'].shift(1)),
                                             (df[f'{col_prefix}High'] - df[f'{col_prefix}Low']),
                                             (df[f'{col_prefix}High'] - df[f'{col_prefix}Close'].shift(1)))
    df[f'{col_prefix}ATR'] = df[f'{col_prefix}True Range'].ewm(span=period, adjust=False).mean()

    df[f'{col_prefix}Plus DI'] = 100 * (df[f'{col_prefix}Plus DM EMA'] / df[f'{col_prefix}ATR'])
    df[f'{col_prefix}Minus DI'] = 100 * (df[f'{col_prefix}Minus DM EMA'] / df[f'{col_prefix}ATR'])
    
    df[f'{col_prefix}DX'] = 100 * (abs(df[f'{col_prefix}Plus DI'] - df[f'{col_prefix}Minus DI']) / 
                                   (df[f'{col_prefix}Plus DI'] + df[f'{col_prefix}Minus DI']))
    df[f'{col_prefix}ADX'] = df[f'{col_prefix}DX'].ewm(span=period, adjust=False).mean()

def calculate_SO(df, identifier):
    df[f'L14_{identifier}'] = df[f'Low_{identifier}'].rolling(window=14).min()
    df[f'H14_{identifier}'] = df[f'High_{identifier}'].rolling(window=14).max()
    df[f'%K_{identifier}'] = 100 * ((df[f'Close_{identifier}'] - df[f'L14_{identifier}']) / (df[f'H14_{identifier}'] - df[f'L14_{identifier}']))
    df[f'%D_{identifier}'] = df[f'%K_{identifier}'].rolling(window=3).mean()

def calculate_CCI(df, identifier):
    TP = (df[f'High_{identifier}'] + df[f'Low_{identifier}'] + df[f'Close_{identifier}']) / 3
    df[f'SMA_{identifier}'] = TP.rolling(window=20).mean()
    df[f'MAD_{identifier}'] = abs(TP - df[f'SMA_{identifier}']).rolling(window=20).mean()
    df[f'CCI_{identifier}'] = (TP - df[f'SMA_{identifier}']) / (0.015 * df[f'MAD_{identifier}'])

def calculate_ATR(df, identifier, windows=[14], columnNameHigh='High', columnNameLow='Low', columnNameClose='Close'):
    """
    Calculate Average True Range (ATR) for cryptocurrency data over multiple windows.

    :param df: DataFrame - The data containing high, low, and close price information.
    :param identifier: str - A unique identifier for the cryptocurrency.
    :param windows: list of int - The list of windows over which to calculate ATR. Default is [14].
    :param columnNameHigh: str - The name of the column with high price data. Default is 'High'.
    :param columnNameLow: str - The name of the column with low price data. Default is 'Low'.
    :param columnNameClose: str - The name of the column with close price data. Default is 'Close'.
    :return: DataFrame - The original data with added columns for ATR for each window in windows.
    """
    high_low = df[f'{columnNameHigh}_{identifier}'] - df[f'{columnNameLow}_{identifier}']
    high_prev_close = abs(df[f'{columnNameHigh}_{identifier}'] - df[f'{columnNameClose}_{identifier}'].shift(1))
    low_prev_close = abs(df[f'{columnNameLow}_{identifier}'] - df[f'{columnNameClose}_{identifier}'].shift(1))
    true_range = pd.DataFrame({
        f'High-Low_{identifier}': high_low,
        f'High-PrevClose_{identifier}': high_prev_close,
        f'Low-PrevClose_{identifier}': low_prev_close
    }).max(axis=1)
    
    new_columns = {}
    for window in windows:
        atr = true_range.ewm(span=window, adjust=False).mean().fillna(0)
        new_columns.update({f'ATR_{window}_{identifier}': atr})
        
    df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
    return df

def calculate_KSI(df, identifier):
    df[f'KSI_{identifier}'] = 100 * ((df[f'Close_{identifier}'] - df[f'Low_{identifier}']) / (df[f'High_{identifier}'] - df[f'Low_{identifier}']))

def calculate_KST(df, identifier):
    """
    Calculate the Know Sure Thing (KST) and its Signal Line for cryptocurrency data.

    The KST is a momentum oscillator that combines four different rate-of-change (ROC) periods 
    into a single indicator. It is typically used to identify major stock market cycle junctures.

    :param df: DataFrame - The data containing Close price information.
    :param identifier: str - A unique identifier for the cryptocurrency.
    :return: DataFrame - The original data with added columns for KST and its Signal Line for the identifier.
    """
    # Convert periods from traditional market days to 1M data for 24/7 market
    periods = [9, 12, 18, 24]  # Original periods in days
    converted_periods = [(period * 390) // 1440 for period in periods]

    # Calculate the rate of change for the converted periods
    ROCs = []
    for i, period in enumerate(converted_periods):
        ROC = ((df[f'Close_{identifier}'] - df[f'Close_{identifier}'].shift(period)) / df[f'Close_{identifier}'].shift(period)) * 100
        ROCs.append(ROC.to_frame(name=f'ROC_{i+1}_{identifier}'))

    # Concatenate all ROC DataFrames
    df = pd.concat([df] + ROCs, axis=1)

    # Calculating the KST (Know Sure Thing) value
    df[f'KST_{identifier}'] = (df[f'ROC_1_{identifier}'] * 1) + (df[f'ROC_2_{identifier}'] * 2) + (df[f'ROC_3_{identifier}'] * 3) + (df[f'ROC_4_{identifier}'] * 4)
    
    # Calculating the KST Signal Line
    df[f'KSTSignalLine_{identifier}'] = df[f'KST_{identifier}'].ewm(span=9, adjust=False).mean()
    
    # Replace infinite and NaN values with 0
    df[f'KST_{identifier}'].replace(np.inf, 0, inplace=True)
    df.fillna(0, inplace=True)
    
    return df

def calculate_MFI(df, identifier):
    # Convert the 14-day period from traditional markets to 1M data for 24/7 market
    converted_period = (14 * 390) // 1440
    
    # Calculating typical price
    typical_price = (df[f'High_{identifier}'] + df[f'Low_{identifier}'] + df[f'Close_{identifier}']) / 3
    
    # Calculating raw money flow
    money_flow = typical_price * df[f'Volume_{identifier}']
    
    # Determine positive and negative money flow
    df[f'PositiveMoneyFlow_{identifier}'] = np.where(typical_price > typical_price.shift(1), money_flow, 0)
    df[f'NegativeMoneyFlow_{identifier}'] = np.where(typical_price < typical_price.shift(1), 0, money_flow)
    
    # Calculating the sum of positive and negative money flow using the converted period
    df[f'PositiveMoneyFlow_{identifier}'] = df[f'PositiveMoneyFlow_{identifier}'].rolling(window=converted_period).sum()
    df[f'NegativeMoneyFlow_{identifier}'] = df[f'NegativeMoneyFlow_{identifier}'].rolling(window=converted_period).sum()
    
    # Calculating Money Ratio
    df[f'MoneyRatio_{identifier}'] = df[f'PositiveMoneyFlow_{identifier}'] / df[f'NegativeMoneyFlow_{identifier}']
    
    # Handling Inf values in Money Ratio
    df[f'MoneyRatio_{identifier}'].replace(np.inf, np.nan, inplace=True)
    
    # Calculating Money Flow Index
    df[f'MoneyFlowIndex_{identifier}'] = 100 - (100 / (1 + df[f'MoneyRatio_{identifier}']))
    
    # Handling NaN and Inf values in MFI
    df[f'MoneyFlowIndex_{identifier}'].replace([np.inf, -np.inf], 0, inplace=True)
    
    # Filling NaN values with 0 as required
    df.fillna(0, inplace=True)
    
    return df

def calculate_OBV(df, identifier):
    # Calculate the OBV
    obv = np.where(df[f'Close_{identifier}'] > df[f'Close_{identifier}'].shift(1), df[f'Volume_{identifier}'],
                   np.where(df[f'Close_{identifier}'] < df[f'Close_{identifier}'].shift(1), -df[f'Volume_{identifier}'], 0))
    df[f'OBV_{identifier}'] = obv.cumsum()
    
    # Replacing NaN values with 0, although they shouldn't typically be produced in OBV calculations
    df.fillna(0, inplace=True)
    
    # Replace infinite values with 0, if any
    df.replace(np.inf, 0, inplace=True)
    df.replace(-np.inf, 0, inplace=True)

    return df

def calculate_ROC(df, identifier):
    df[f'ROC_{identifier}'] = ((df[f'Close_{identifier}'] - df[f'Close_{identifier}'].shift(12)) / df[f'Close_{identifier}'].shift(12)) * 100

def calculate_StochRSI(df, identifier):
    df[f'RSI_{identifier}'] = df[f'Close_{identifier}'].rolling(window=14).mean()
    df[f'RSI_{identifier}'] = np.where(df[f'RSI_{identifier}'] == 0, 100, df[f'RSI_{identifier}'])
    df[f'RSI_{identifier}'] = np.where(df[f'RSI_{identifier}'] == 100, 0, df[f'RSI_{identifier}'])
    delta = df[f'Close_{identifier}'].diff(1)
    up = delta.where(delta > 0, 0)
    down = -delta.where(delta < 0, 0)
    avg_gain = up.rolling(window=14).mean()
    avg_loss = down.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df[f'StochRSI_{identifier}'] = 100 * ((rs - rs.min()) / (rs.max() - rs.min()))

def calculate_TSI(df, identifier):
    df[f'PC_{identifier}'] = df[f'Close_{identifier}'] - df[f'Close_{identifier}'].shift(1)
    df[f'ABS_PC_{identifier}'] = abs(df[f'PC_{identifier}'])
    df[f'EMA_25_PC_{identifier}'] = df[f'PC_{identifier}'].ewm(span=25, adjust=False).mean()
    df[f'EMA_13_ABS_PC_{identifier}'] = df[f'ABS_PC_{identifier}'].ewm(span=13, adjust=False).mean()
    df[f'TSI_{identifier}'] = 100 * (df[f'EMA_25_PC_{identifier}'] / df[f'EMA_13_ABS_PC_{identifier}'])
    df[f'Signal Line_{identifier}'] = df[f'TSI_{identifier}'].ewm(span=13, adjust=False).mean()

def calculate_UO(df, identifier):
    df[f'BP_{identifier}'] = df[f'Close_{identifier}'] - np.where(df[f'Close_{identifier}'] < df[f'Low_{identifier}'].shift(1), df[f'Close_{identifier}'], 
                                                                  np.where(df[f'Close_{identifier}'] > df[f'High_{identifier}'].shift(1), df[f'High_{identifier}'].shift(1), df[f'Low_{identifier}'].shift(1)))
    df[f'TR_{identifier}'] = np.where(df[f'Close_{identifier}'] < df[f'Low_{identifier}'].shift(1), df[f'High_{identifier}'].shift(1) - df[f'Low_{identifier}'].shift(1), 
                                      np.where(df[f'Close_{identifier}'] > df[f'High_{identifier}'].shift(1), df[f'High_{identifier}'].shift(1) - df[f'Low_{identifier}'].shift(1), 
                                               df[f'High_{identifier}'].shift(1) - df[f'Low_{identifier}'].shift(1)))
    df[f'Average7_{identifier}'] = df[f'BP_{identifier}'].rolling(window=7).sum() / df[f'TR_{identifier}'].rolling(window=7).sum()
    df[f'Average14_{identifier}'] = df[f'BP_{identifier}'].rolling(window=14).sum() / df[f'TR_{identifier}'].rolling(window=14).sum()
    df[f'Average28_{identifier}'] = df[f'BP_{identifier}'].rolling(window=28).sum() / df[f'TR_{identifier}'].rolling(window=28).sum()
    df[f'UO_{identifier}'] = 100 * ((4 * df[f'Average7_{identifier}']) + (2 * df[f'Average14_{identifier}']) + df[f'Average28_{identifier}']) / (4 + 2 + 1)
    df[f'Signal Line_{identifier}'] = df[f'UO_{identifier}'].rolling(window=3).mean()

def calculate_VI(df, identifier):
    # Calculate Positive and Negative Vortex Movements with conditions
    df[f'VM+_{identifier}'] = np.where(df[f'High_{identifier}'] < df[f'Low_{identifier}'].shift(1), 0, 
                                       abs(df[f'High_{identifier}'] - df[f'Low_{identifier}'].shift(1)))

    df[f'VM-_{identifier}'] = np.where(df[f'Low_{identifier}'] > df[f'High_{identifier}'].shift(1), 0, 
                                       abs(df[f'Low_{identifier}'] - df[f'High_{identifier}'].shift(1)))

    # Calculate the 14-period rolling sum for VM+ and VM-
    df[f'VM+14_{identifier}'] = df[f'VM+_{identifier}'].rolling(window=4858).sum()
    df[f'VM-14_{identifier}'] = df[f'VM-_{identifier}'].rolling(window=4858).sum()

    # Calculate Positive and Negative Vortex Indicators
    df[f'VI+_{identifier}'] = df[f'VM+14_{identifier}'] / df[f'VM-14_{identifier}']
    df[f'VI-_{identifier}'] = df[f'VM-14_{identifier}'] / df[f'VM+14_{identifier}']

    # Handle infinite values
    df[f'VI+_{identifier}'].replace([np.inf, -np.inf], 100, inplace=True)
    df[f'VI-_{identifier}'].replace([np.inf, -np.inf], 100, inplace=True)

    # Calculate Vortex Indicator and its 6-period rolling mean
    df[f'VI_{identifier}'] = abs(df[f'VI+_{identifier}'] - df[f'VI-_{identifier}']).rolling(window=6).mean()
    
    # Fill NaN values
    df.fillna(0, inplace=True)
    
    return df

def calculate_WR(df, identifier):
    df[f'%R_{identifier}'] = ((df[f'High_{identifier}'].rolling(window=14).max() - df[f'Close_{identifier}']) / 
                              (df[f'High_{identifier}'].rolling(window=14).max() - df[f'Low_{identifier}'].rolling(window=14).min())) * -100

def calculate_ADI(df, identifier):
    df[f'MF Multiplier_{identifier}'] = ((df[f'Close_{identifier}'] - df[f'Low_{identifier}']) - (df[f'High_{identifier}'] - df[f'Close_{identifier}'])) / (df[f'High_{identifier}'] - df[f'Low_{identifier}'])
    df[f'MF Volume_{identifier}'] = df[f'MF Multiplier_{identifier}'] * df[f'Volume_{identifier}']
    df[f'ADI_{identifier}'] = df[f'MF Volume_{identifier}'].cumsum()

def calculate_CMF(df, identifier):
    df[f'MF Multiplier_{identifier}'] = ((df[f'Close_{identifier}'] - df[f'Low_{identifier}']) - (df[f'High_{identifier}'] - df[f'Close_{identifier}'])) / (df[f'High_{identifier}'] - df[f'Low_{identifier}'])
    df[f'MF Volume_{identifier}'] = df[f'MF Multiplier_{identifier}'] * df[f'Volume_{identifier}']
    df[f'CMF_{identifier}'] = df[f'MF Volume_{identifier}'].rolling(window=20).sum() / df[f'Volume_{identifier}'].rolling(window=20).sum()

def calculate_AI(df, identifier):
    df[f'Up_{identifier}'] = df[f'High_{identifier}'].rolling(window=25).apply(lambda x: x.argmax(), raw=True) / 25 * 100
    df[f'Down_{identifier}'] = df[f'Low_{identifier}'].rolling(window=25).apply(lambda x: x.argmin(), raw=True) / 25 * 100

def calculate_AroonOsc(df, identifier):
    """
    Calculate the Aroon Oscillator (AO) for cryptocurrency data.

    The Aroon Oscillator is calculated by subtracting the Aroon Down from the Aroon Up. 
    Aroon Up and Down are measures of how long it has been since the highest high/lowest low 
    over a past period (typically 25 days).

    :param df: DataFrame - The data containing High and Low price information.
    :param identifier: str - A unique identifier for the cryptocurrency.
    :return: DataFrame - The original data with added columns for Aroon Up, Aroon Down, and AO for the identifier.
    """
    window = 25

    # Calculate the rolling max and min for the high and low prices
    rolling_max = df[f'High_{identifier}'].rolling(window=window).max()
    rolling_min = df[f'Low_{identifier}'].rolling(window=window).min()

    # Identify the location (index) of the rolling max and min
    up = df[f'High_{identifier}'].rolling(window=window).apply(lambda x: np.where(x == rolling_max[x.index[-1]])[0][-1], raw=False)
    down = df[f'Low_{identifier}'].rolling(window=window).apply(lambda x: np.where(x == rolling_min[x.index[-1]])[0][-1], raw=False)

    # Normalize and scale the values to a percentage
    df[f'Up_{identifier}'] = ((window - 1) - up) / (window - 1) * 100
    df[f'Down_{identifier}'] = ((window - 1) - down) / (window - 1) * 100

    # Calculate the Aroon Oscillator
    df[f'AroonOsc_{identifier}'] = df[f'Up_{identifier}'] - df[f'Down_{identifier}']

    return df

def calculate_BoP(df, identifier):
    df[f'BoP_{identifier}'] = (df[f'Close_{identifier}'] - df[f'Open_{identifier}']) / (df[f'High_{identifier}'] - df[f'Low_{identifier}'])

def calculate_CG(df, identifier):
    df[f'CG_{identifier}'] = (df[f'High_{identifier}'] + df[f'Low_{identifier}']) / 2

def calculate_CMO(df, identifier):
    df[f'CMO_{identifier}'] = ((df[f'Close_{identifier}'] - df[f'Close_{identifier}'].rolling(window=20).mean()) / 
                               (df[f'Close_{identifier}'] + df[f'Close_{identifier}'].rolling(window=20).mean())) * 100

def calculate_CC(df, identifier):
    df[f'ROC_14_{identifier}'] = ((df[f'Close_{identifier}'] - df[f'Close_{identifier}'].shift(14)) / df[f'Close_{identifier}'].shift(14)) * 100
    df[f'ROC_11_{identifier}'] = ((df[f'Close_{identifier}'] - df[f'Close_{identifier}'].shift(11)) / df[f'Close_{identifier}'].shift(11)) * 100
    df[f'CC_{identifier}'] = df[f'ROC_14_{identifier}'] + df[f'ROC_11_{identifier}']
    df[f'Signal Line_{identifier}'] = df[f'CC_{identifier}'].ewm(span=10, adjust=False).mean()

def calculate_DPO(df, identifier):
    df[f'DPO_{identifier}'] = df[f'Close_{identifier}'].shift(int((0.5 * 20) + 1)) - df[f'Close_{identifier}'].rolling(window=20).mean()

def calculate_EoM(df, identifier):
    df[f'Midpoint_{identifier}'] = (df[f'High_{identifier}'] + df[f'Low_{identifier}']) / 2
    df[f'Box Ratio_{identifier}'] = (df[f'Volume_{identifier}'] / 100000000) / (df[f'High_{identifier}'] - df[f'Low_{identifier}'])
    df[f'EoM_{identifier}'] = df[f'Midpoint_{identifier}'].diff(1) / df[f'Box Ratio_{identifier}'].diff(1)

def calculate_FI(df, identifier):
    df[f'FI_{identifier}'] = df[f'Close_{identifier}'].diff(1) * df[f'Volume_{identifier}']

def calculate_forecast_error(df, identifier, forecast_horizons=[1, 5, 15, 30, 60, 120]):
    for horizon in forecast_horizons:
        df[f'ForecastError_{horizon}min_{identifier}'] = abs(df[f'ForecastedClose_{horizon}min_{identifier}'] - df[f'Close_{identifier}'])
    return df

def calculate_max_return(df, identifier, time_period='24h', strategy='long', forecast_step=1):
    if time_period == '24h':
        window_size = 24 * 60 // forecast_step  # Adjust window size based on the forecast step
    else:
        # Adjust window_size based on the specified time_period and forecast_step
        pass
    
    if strategy == 'long':
        df[f'MaxReturn_{time_period}_{identifier}'] = df[f'ForecastedClose_{forecast_step}step_{identifier}'].rolling(window=window_size).apply(lambda x: x.max() - x.min())
    elif strategy == 'short':
        df[f'MaxReturn_{time_period}_{identifier}'] = df[f'ForecastedClose_{forecast_step}step_{identifier}'].rolling(window=window_size).apply(lambda x: x.min() - x.max())
    elif strategy == 'market_making':
        df[f'MaxReturn_{time_period}_{identifier}'] = df[f'ForecastedClose_{forecast_step}step_{identifier}'].rolling(window=window_size).apply(lambda x: max(x.max() - x.min(), x.min() - x.max()))
    
    return df

def calculate_optimal_trade(df, identifier, time_period='24h', strategy='long', forecast_step=1):
    if time_period == '24h':
        window_size = 24 * 60 // forecast_step  # Adjust window size based on the forecast step
    else:
        # Adjust window_size based on the specified time_period and forecast_step
        pass
    
    if strategy == 'long':
        df[f'OptimalTrade_{time_period}_{identifier}'] = df[f'ForecastedClose_{forecast_step}step_{identifier}'].rolling(window=window_size).apply(lambda x: x.idxmax() - x.idxmin())
    elif strategy == 'short':
        df[f'OptimalTrade_{time_period}_{identifier}'] = df[f'ForecastedClose_{forecast_step}step_{identifier}'].rolling(window=window_size).apply(lambda x: x.idxmin() - x.idxmax())
    elif strategy == 'market_making':
        df[f'OptimalTrade_{time_period}_{identifier}'] = df[f'ForecastedClose_{forecast_step}step_{identifier}'].rolling(window=window_size).apply(lambda x: x.idxmax() if x.max() - x.min() > x.min() - x.max() else x.idxmin())
    
    return df