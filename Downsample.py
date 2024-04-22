import logging
import pandas as pd
import os
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import argparse
import pytz


def split_identifier(file_path, split_char='_', index=0):
    # Extract the base filename without path
    base_name = os.path.basename(file_path)
    # Remove the file extension to get the raw identifier
    raw_identifier = os.path.splitext(base_name)[0]
    # Split by the specified character and return the desired index
    return raw_identifier.split(split_char)[index]

def process_individual_ticker(file_path, output_dir, split_char='_', index=0, aggregate_window=None):
    """
    Processes an individual ticker's data from a given CSV file path.
    It involves converting UNIX MS timestamps to a readable datetime format and
    resampling the data to a specified interval (downsampling) if specified.

    :param file_path: Path to the CSV file containing the ticker's data.
    :param output_dir: Directory to save the downsampled data.
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
            
            # Convert UNIX MS timestamp to datetime with UTC timezone
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms', utc=True)
            
            # Set 'Timestamp' as the DataFrame index
            df.set_index('Timestamp', inplace=True)
            
            # Define the aggregation methods for OHLCV (Open, High, Low, Close, Volume) data
            agg_methods = {
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum',
                'VWAP': 'mean'
            }
            
            # Resample the DataFrame according to the specified aggregate window
            # and aggregate using the defined methods
            df_resampled = df.resample(aggregate_window).agg(agg_methods)
            
            # Forward fill NaN values for periods with no trades
            df_resampled.ffill(inplace=True)
            
            # Reset index to convert 'Timestamp' back into a column
            df_resampled.reset_index(inplace=True)
            
            # Update the original DataFrame with the resampled data
            df = df_resampled
        
        last_record_time = df['Timestamp'].iloc[-1].strftime('%Y%m%d_%H%M%S')
        
        # Check if the aggregate_window is specified to determine the filename format
        if aggregate_window is None:
            print("No aggregate window specified. Saving to default filename.")
            output_filename = f'{ticker}_{last_record_time}.csv'
        else:
            output_filename = f'{ticker}_{aggregate_window}_{last_record_time}.csv'
        
        # Create the full output path
        output_path = os.path.join(output_dir, output_filename)
        
        # Save the DataFrame to CSV
        df.to_csv(output_path, index=False)
        
        logging.info(f"Saved downsampled data for {ticker} at {aggregate_window} to {output_path}")

    except Exception as e:
        logging.exception("An error occurred: %s", e)
        # Log Traceback
        logging.exception("Traceback: %s", e.__traceback__)

def main(identifier=None, aw='15T'):
    # Define the directory path
    directory_path = './Data'
    
    # Check if the directory exists
    if not os.path.exists(directory_path):
        logging.error("Data directory not found. Please ensure the data is available in the 'Data' directory.")
        return
    
    # Create the directory for downsampled data if it doesn't exist
    downsampled_directory = os.path.join(directory_path, 'Downsampled')
    os.makedirs(downsampled_directory, exist_ok=True)
    
    if identifier:
        # Process a specific CSV file based on the identifier
        file_path = os.path.join(directory_path, f"{identifier}.csv")
        if os.path.exists(file_path):
            process_individual_ticker(file_path, downsampled_directory, aggregate_window=aw)
            
            # Save the downsampled data to the specified file path
            downsampled_file_path = os.path.join(downsampled_directory, f"{identifier}_downsampled.csv")
            logging.info(f"Downsampled data for {identifier} saved to {downsampled_file_path}.")
        else:
            logging.error(f"File not found for identifier: {identifier}")
    else:
        # Get a list of all CSV files in the directory
        csv_files = [str(file) for file in Path(directory_path).glob('*.csv')]
        
        # Process each CSV file
        for file_path in tqdm(csv_files):
            process_individual_ticker(file_path, downsampled_directory, aggregate_window=aw)
            
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create an argument parser
    parser = argparse.ArgumentParser(description="Downsample and process CSV files.")
    parser.add_argument("-i", "--identifier", help="Identifier of the specific CSV file to process.")
    parser.add_argument("-aw", "--aggregate_window", default='15T', help="Aggregate window in XT format (default: 15T).")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug logging.")

    # Parse the arguments
    args = parser.parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("Debug logging enabled.")

    main(identifier=args.identifier, aw=args.aggregate_window)