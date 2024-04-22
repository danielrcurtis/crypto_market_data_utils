from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from functools import reduce
import numpy as np
import os
import argparse
import TaCalcs as ta
import SignalDataAnalysis as sda
import datetime
import logging
import multiprocessing

def load_csv_from_directory(directory_path='./data/', return_dataframe=False):
    """
    Load all .csv files from the specified directory into pandas DataFrames.

    :param directory_path: Path to the directory to read from. Defaults to './data/'.
    :return: List of pandas DataFrames.
    """
    # Ensure the directory exists
    if not os.path.exists(directory_path):
        print(f"The directory '{directory_path}' does not exist!")
        return []

    # List all files in the directory
    all_files = os.listdir(directory_path)

    # Filter out files that don't end with .csv
    csv_files = [file for file in all_files if file.endswith('.csv')]

    # Create an empty list to store DataFrames
    dfs = []

    if return_dataframe:
        # Loop through each .csv file and read it into a DataFrame
        for file in csv_files:
            logging.debug(f"Reading file: {file}")
            file_path = os.path.join(directory_path, file)
            df = pd.read_csv(file_path)
            dfs.append(df)
        return dfs
    else:
        for file in csv_files:
            file_path = os.path.join(directory_path, file)
            dfs.append(file_path)
        return dfs

def load_data_for_tickers(tickers):
    dataframes = []
    for ticker in tickers:
        path = f"./data/{ticker}.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            if not df.empty:
                dataframes.append(df)
            else:
                logging.warning(f"{ticker}.csv is empty.")
        else:
            logging.warning(f"{ticker}.csv does not exist.")
    return dataframes

def create_enhanced_dataset(df, identifier, strategies=['long', 'short', 'market_making']):
    # Add historical price forecasts
    # ...

    # Calculate cumulative forecast error per step
    df = sda.calculate_forecast_error(df, identifier)

    for strategy in strategies:
        # Calculate maximum return within the specified time period for each strategy
        df = sda.calculate_max_return(df, identifier, strategy=strategy)

        # Calculate optimal trade values within the specified time period for each strategy
        df = sda.calculate_optimal_trade(df, identifier, strategy=strategy)

    return df

def process_multiple_identifiers(dfs, identifiers, strategies=['long', 'short', 'market_making']):
    enhanced_dfs = []
    for df, identifier in zip(dfs, identifiers):
        enhanced_df = create_enhanced_dataset(df, identifier, strategies)
        enhanced_dfs.append(enhanced_df)
    
    return enhanced_dfs

# Define a function to split the identifier from a file path
def split_identifier(file_path, split_char='_', index=0):
    return file_path.split('/')[-1].split(split_char)[index]

def build_aggregate_dataframe(csv_files, identifier):
    dfs = []
    # Read all files into DataFrames and store in a list with renamed columns
    for file, identifier in zip(csv_files, identifier):
        df = pd.read_csv(file)
        logging.info(f"Read DataFrame for {file} with shape: {df.shape}")

        # Renaming columns
        df.columns = [f"{col}_{identifier}" if col != "Time" else "Time" for col in df.columns]
        dfs.append(df)
    return dfs

# Define a function to merge two DataFrames on 'Time'
def merge_dfs(left, right):
    return pd.merge(left, right, on='Time')

# Build a dataset with technical indicators
def build_dataset(dfs, output_filename):
    if not dfs:  # Check if dfs is empty
        logging.error("Empty dataframes list passed to build_dataset. Aborting...")
        return
    if logging.info:
        dfs[0].describe()
        logging.info(f"Number of DataFrames: {len(dfs)}")
        logging.info(f"Number of columns in each DataFrame: {len(dfs[0].columns)}")
        logging.info(f"Columns in each DataFrame: {dfs[0].columns}")
        logging.info(f"Output filename: {output_filename}")
   
    # Calculate the technical indicators for each DataFrame 
    for df in dfs:
        if logging.info:
            logging.info(f"Calculating technical indicators for DataFrame: {df}")
        identifier = df.columns[1].split("_")[1]
        ta.calculate_SMA(df, identifier)
        ta.calculate_EMA(df, identifier)
        ta.calculate_KST(df, identifier)
        ta.calculate_MFI(df, identifier)
        ta.calculate_VI(df, identifier)
        ta.calculate_OBV(df, identifier)
        ta.calculate_RSI(df, identifier)
        
    # Use the reduce function to successively merge each DataFrame
    merged_df = reduce(merge_dfs, dfs)

    # Drop columns with only NaN values
    merged_df.dropna(axis=1, how='all', inplace=True)

    # Save the merged dataframe to a new CSV file
    merged_df.to_csv(output_filename, index=False)

    print(f"Saved aggregate TA data to {output_filename}")

# Define a function to build a dataset with technical indicators for a single stock ticker
def build_dataset_for_single_ticker(df, ticker, output_filename):
    """
    Aggregates financial time series data for a specific stock ticker and saves the aggregated data to a CSV file.

    This function processes a DataFrame containing time series data for a stock, aggregates it based on a specified 
    time window, and calculates OHLC (Open-High-Low-Close) values along with the sum of volumes and transactions.
    It then saves the aggregated data to a specified CSV file.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the financial time series data.
    ticker (str): The stock ticker symbol for which the dataset is being built.
    output_filename (str): The name of the output file where the aggregated data will be saved.

    Returns:
    None: The function does not return anything but saves the aggregated data to a CSV file.
    """
    try:
        # Describe the DataFrame for logging purposes
        if logging.info:
            df.describe()  # Generate descriptive statistics of the DataFrame
            # Log the number of columns, column names, and the output file name
            logging.info(f"Number of columns in the DataFrame: {len(df.columns)}")
            logging.info(f"Columns in the DataFrame: {df.columns}")
            logging.info(f"Output filename: {output_filename}")

        # Log the beginning of technical indicator calculations
        if logging.info:
            logging.info(f"Calculating technical indicators for DataFrame: {df}")




        ta.calculate_SMA(df, ticker)
        ta.calculate_EMA(df, ticker)
        ta.calculate_SMA(df, ticker, columnName='High')
        ta.calculate_SMA(df, ticker, columnName='Low')
        ta.calculate_EMA(df, ticker, columnName='High')
        ta.calculate_EMA(df, ticker, columnName='Low')
        ta.calculate_return(df, ticker, periods=[1, 5, 11, 22, 60])
        ta.calculate_price_roc(df, ticker, periods=[1, 5, 11, 22, 60])
        ta.calculate_volume_change(df, ticker, periods=[1, 5, 11, 22, 60])
        ta.calculate_volume_roc(df, ticker, ns=[10, 30, 60, 120, 240])
        ta.calculate_historical_volatility(df, ticker, windows=[10, 30, 60, 120, 240])
        ta.calculate_BB(df, ticker, windows=[10, 20, 60, 120, 240])
        ta.calculate_ATR(df, ticker, windows=[10, 14, 20, 60, 120, 240])
        ta.calculate_AO(df, ticker)
        ta.calculate_DonchianChannels(df, ticker)
        ta.calculate_KST(df, ticker)
        ta.calculate_MFI(df, ticker)
        ta.calculate_VI(df, ticker)
        ta.calculate_OBV(df, ticker)
        ta.calculate_RSI(df, ticker)
        ta.calculate_MACD_Dataset(df, ticker, 8, 17, 9)
        ta.calculate_MACD_Dataset(df, ticker, 12, 26, 9)
        ta.calculate_MACD_Dataset(df, ticker, 5, 21, 9)
        ta.calculate_MACD_Dataset(df, ticker, 3, 10, 4)
        ta.calculate_MACD_Dataset(df, ticker, 11, 22, 9)
        ta.calculate_MACD_Dataset(df, ticker, 101, 224, 49)
        ta.calculate_VWAP(df, ticker)
        ta.append_rolling_outlier_signal(df=df, identifier=ticker, window_size=5, multiplier=1.5)
        ta.append_rolling_outlier_signal(df=df, identifier=ticker, window_size=10, multiplier=1.5)
        ta.append_rolling_outlier_signal(df=df, identifier=ticker, window_size=15, multiplier=1.5)
        ta.append_rolling_outlier_signal(df=df, identifier=ticker, window_size=30, multiplier=1.5)
        ta.calculate_price_change(df, ticker)
        ta.calculate_price_roc(df, ticker)
        ta.calculate_volume_change(df, ticker)
        ta.calculate_volume_roc(df, ticker)
        ta.calculate_historical_volatility(df, ticker)
        ta.calculate_return(df, ticker)
            
        # Drop columns with only NaN values
        df.dropna(axis=1, how='all', inplace=True)

        logging.info(f"Saved TA data for {ticker} to {output_filename}")
    except Exception as e:
        logging.exception("An error occurred: %s", e)
        # Log Traceback
        logging.exception("Traceback: %s", e.__traceback__)

def process_individual_ticker(file_path, split_char='_', index=0, aggregate_window=None):
    """
    Processes an individual ticker's data from a given CSV file path. It involves converting UNIX MS timestamps
    to a readable datetime format and resampling the data to a specified interval (downsampling) if specified.

    :param file_path: Path to the CSV file containing the ticker's data.
    :param split_char: Character used to split the file name to extract the ticker name. Default is '_'.
    :param index: Index position of the ticker name in the split file name. Default is 0.
    :param aggregate_window: String representing the new sampling rate, e.g., '5T' for 5 minutes, '15T' for 15 minutes, etc.
    """

    try:
        logging.info(f"Starting processing for file: {file_path}")

        # Extract the ticker name from the file path
        ticker = split_identifier(file_path, split_char, index)
        logging.info(f"Processing ticker: {ticker}")

        # Load the data from the CSV file
        df = pd.read_csv(file_path)

        if aggregate_window is not None:
            logging.info(f"Downsampling data to {aggregate_window}...")

            # Convert UNIX MS timestamp to datetime
            df['Time'] = pd.to_datetime(df['Time'], unit='ms')

            # Set 'Time' as the DataFrame index
            df.set_index('Time', inplace=True)

            # Define the aggregation methods for OHLCV (Open, High, Low, Close, Volume) data
            agg_methods = {
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }

            # Resample the DataFrame according to the specified aggregate window
            # and aggregate using the defined methods
            df_resampled = df.resample(aggregate_window).agg(agg_methods)

            # Forward fill NaN values for periods with no trades
            df_resampled.ffill(inplace=True)

            # Reset index to convert 'Time' back into a column
            df_resampled.reset_index(inplace=True)

            # Update the original DataFrame with the resampled data
            df = df_resampled

        # Rename columns by appending the ticker name
        df.rename(columns=lambda x: f"{x}_{ticker}" if x != 'Time' else x, inplace=True)

        current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        if aggregate_window is None:
            print("No aggregate window specified. Saving to default filename.")
            output_filename = os.path.join('./datasets', f'{ticker}_ta_{current_time}.csv')
        else:
            output_filename = os.path.join('./datasets', f'{ticker}_ta_dsaw-{aggregate_window}_{current_time}.csv')

       # build_dataset_for_single_ticker(df, ticker, output_filename)

        df.to_csv(output_filename, index=False)

        logging.info(f"Saved downsampled TA data for {ticker} at {aggregate_window} to {output_filename}")
    except Exception as e:
        logging.exception("An error occurred: %s", e)
        # Log Traceback
        logging.exception("Traceback: %s", e.__traceback__)

def process_tickers(csv_files, tickers, split_char, index, output_dir, aggregate=False):
    """
    Processes a list of tickers and saves the results to an output file.

    Parameters:
    - csv_files: List of CSV file paths.
    - tickers: List of tickers to process.
    - split_char: Character to split the identifier on.
    - index: Index of the identifier after splitting.
    - output_dir: Directory to save the output file.
    - aggregate: Whether to aggregate the tickers or not. Default is False.

    Returns:
    - output_filename: The path of the file where the results were saved.
    """
    try:
        if aggregate:
            logging.info(f"Processing and aggregating tickers: {tickers}")
        else:
            logging.info(f"Processing tickers: {tickers}")

        # Filter based on provided tickers
        csv_files = [f for f in csv_files if split_identifier(f, split_char, index) + '.csv' in tickers]

        logging.debug(f"Filtered CSV files: {csv_files}")

        # Build the aggregate DataFrame
        identifiers = [split_identifier(file_path, split_char, index) for file_path in csv_files]
        dfs = build_aggregate_dataframe(csv_files, identifiers)

        # Compute technical indicators and merge dataframes
        output_filename = os.path.join(output_dir, '-'.join(tickers) + '_with_ta.csv')
        build_dataset(dfs, output_filename)
        
        logging.info(f"Data saved to: {output_filename}")
        
        return output_filename
    except Exception as e:
        logging.exception("An error occurred: %s", e)
        return None

def parse_arguments():
    """
    Parses command-line arguments and returns them.

    Returns:
    - args: Parsed arguments.
    """
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Create datasets based on input parameters; the default creates datasets for each ticker available in the ./data/ directory.')

    # Add arguments to the parser
    parser.add_argument('-d', '--directory', type=str, default='./data/',
                        help='Path to the directory to read from. Defaults to "./data/".')
    parser.add_argument('-s', '--split_char', type=str, default='_',
                        help='Character to split the identifier on. Defaults to "_".')
    parser.add_argument('-i', '--index', type=int, default=0,
                        help='Index of the identifier after splitting. Defaults to 0.')
    parser.add_argument('-c', '--columns', type=str, default='Time,Open,High,Low,Close,Volume',
                        help='Comma-separated list of column names. Defaults to "Time,Open,High,Low,Close,Volume".')
    parser.add_argument('-o', '--output', type=str, default='merged.csv',
                        help='Name of the output file. Defaults to "merged.csv".')
    parser.add_argument('-it', '--input_tickers', type=str, default='',
                        help='Name of the files containing the tickers. Defaults to "".')
    parser.add_argument('-df', '--dataframe', type=bool, default=False,
                        help='Whether to return a list of DataFrames or a single merged DataFrame. Defaults to False.')
    parser.add_argument('-g', '--get_tickers', type=bool, default=False,
                        help='Whether to return a list of available tickers or not. Defaults to False.')
    parser.add_argument('-mp', '--max_parallel', type=int, default=1,
                        help='Maximum number of parallel processes. Defaults to 1.')
    parser.add_argument('-ft', '--filtered_tickers_file', type=str, default='',
                        help='Filename containing the list of filtered tickers. Defaults to empty string.')
    parser.add_argument('-ag', '--aggregate', type=bool, default=True,
                        help='Whether to aggregate tickers into a single dataset or not. Defaults to True. For use with the --ft flag.')
    parser.add_argument('-frc', '--filter_by_row_count', type=float, default=None,
                        help='Filters tickers based on row count deviation. Takes in percentage deviation as argument.')
    parser.add_argument('-aw', '--aggregate_window', type=str, default=None,
                        help='Downsamples the data to the specified window. Takes in a string representing the new sampling rate, e.g., "5T" for 5 minutes, "15T" for 15 minutes, etc.')
    parser.add_argument('-mc', '--model_configs', type=str, default='',
                        help='Model configurations for running predictions. Each configuration should be provided as a comma-separated string: model_path,input_columns,output_column,time_steps. Multiple configurations can be separated by a semicolon (;).')

    # Parse the arguments
    args = parser.parse_args()

    return args

def get_row_count(file_path):
    """
    Reads a CSV file and returns its row count.
    """
    df = pd.read_csv(file_path)
    return {file_path: len(df)}

def filter_files_by_row_count(directory_path='./data/', percentage_threshold=0.1):
    """
    Filters files based on the total number of rows in relation to the file with the most rows.

    :param directory_path: Path to the directory where CSV datasets reside.
    :param percentage_threshold: Percentage threshold for filtering the data. Represents the maximum deviation from the file with the most rows.
    :return: List of file names that fit within the percentage range.
    """
    # Load all csv file paths from the directory
    csv_files = load_csv_from_directory(directory_path)

    # Use multiprocessing to calculate row counts
    pool = multiprocessing.Pool(processes=4)
    results = pool.map(get_row_count, csv_files)
    pool.close()
    pool.join()

    # Combine results into a single dictionary
    row_counts = {k: v for r in results for k, v in r.items()}

    # Calculate the maximum row count
    max_row_count = max(row_counts.values())

    # Calculate the lower bound based on the percentage threshold
    lower_bound = max_row_count * (1 - percentage_threshold)

    # Filter the files based on the lower bound
    filtered_files = [file for file, count in row_counts.items() if count >= lower_bound]

    # Save the tickers to a file
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"./filtered_tickers_{current_time}_deviation_{percentage_threshold}.txt"
    with open(filename, 'w') as f:
        for file in filtered_files:
            f.write(f"{file}\n")  # Writing full file path

    return filename  # Return the filename for further use

def main():
    try:
        # Get the parsed arguments
        args = parse_arguments()
        logging.info("Initializing the process...")
        
        # Create the 'datasets' directory if it doesn't exist
        output_dir = './Datasets/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Created output directory: {output_dir}")

        csv_files = load_csv_from_directory(args.directory, return_dataframe=False)
        logging.info(f"Loaded {len(csv_files)} CSV files from the directory: {args.directory}")

        logging.debug(f"Original CSV files: {csv_files}")
        if len(csv_files) == 0:
            logging.warning("No CSV files found in the specified directory!")
            exit()

        # Check if the filter_by_row_count flag is set
        if args.filter_by_row_count is not None:
            filtered_tickers_filename = filter_files_by_row_count(directory_path=args.directory, 
                                                                   percentage_threshold=args.filter_by_row_count)
            # Update the args.filtered_tickers_file to the returned filename
            args.filtered_tickers_file = filtered_tickers_filename
            logging.info(f"Filtered tickers saved to: {filtered_tickers_filename}")

        if args.get_tickers:
            identifiers = [split_identifier(file_path, args.split_char, args.index) for file_path in csv_files]
            formatted_output = ','.join(identifiers)
            print(formatted_output)
            logging.info(f"Identifiers extracted: {formatted_output}")
            exit()

        # Check if the aggregate flag is set
        print(args.aggregate_window)

        if args.input_tickers:
            tickers = args.input_tickers.split(',')
        else:
            tickers = None

        # Explicitly check for the --ft flag
        if args.filtered_tickers_file:
            # Input validation
            if not os.path.exists(args.filtered_tickers_file):
                logging.error(f"The file specified by --ft does not exist: {args.filtered_tickers_file}")
                exit()

            with open(args.filtered_tickers_file, 'r') as f:
                tickers = [line.strip() for line in f.readlines() if line.strip()]

            logging.info(f"Using filtered tickers from file: {args.filtered_tickers_file}")
            args.aggregate = True  # Set aggregate to True if --ft is used.

        # Check if the --model_configs flag is set
        if args.model_configs:
            # Parse the model configurations from the command-line argument
            model_configs = []
            for config_str in args.model_configs.split(';'):
                config_parts = config_str.split(',')
                model_config = {
                    'model_path': config_parts[0],
                    'input_columns': config_parts[1].split(':'),
                    'output_column': config_parts[2],
                    'time_steps': int(config_parts[3])
                }
                model_configs.append(model_config)

            # Process each ticker and run model predictions
            for ticker in tickers:
                logging.info(f"Processing ticker: {ticker}")
                df = pd.read_csv(f"./data/{ticker}.csv")
                enhanced_df = sda.run_model_predictions(df, ticker, model_configs)
                output_filename = os.path.join(output_dir, f"{ticker}_with_model_predictions.csv")
                enhanced_df.to_csv(output_filename, index=False)
                logging.info(f"Saved enhanced data for {ticker} with model predictions to {output_filename}")

        else:
            try:
                if tickers == None:
                    logging.info("Processing individual tickers using parallel processing...")
                    with ProcessPoolExecutor(max_workers=args.max_parallel) as executor:
                        executor.map(process_individual_ticker, csv_files, [args.split_char] * len(csv_files), [args.index] * len(csv_files), [args.aggregate_window] * len(csv_files)) 
                    logging.info("Processing completed for all individual tickers.")
                    return
                if tickers and args.aggregate:
                    process_tickers(csv_files, tickers, args.split_char, args.index, output_dir, aggregate=True)

                elif tickers:
                    process_tickers(csv_files, tickers, args.split_char, args.index, output_dir)
            except Exception as e:
                logging.exception("An error occurred: %s", e)

    except Exception as e:
        logging.exception("An error occurred: %s", e)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    if logging.debug:
        logging.debug("Debug logging enabled.")
    main()