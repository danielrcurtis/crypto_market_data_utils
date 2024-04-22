from DataLoader import fetch_and_save_data
import pandas as pd

def read_tickers_from_csv(filename="tickers.csv"):
    """
    Reads tickers from a CSV file where tickers are comma-separated. 

    Args:
        filename (str): The path to the tickers CSV file. Defaults to 'tickers.csv'.

    Returns:
        list: A list of tickers read from the CSV file.
    """
    # Read the CSV file as a pandas DataFrame
    df = pd.read_csv(filename, header=None, sep=',\s*|\s*,\s*', engine='python')  # added a regex separator to remove any unwanted spaces
    
    # Convert the DataFrame to a list
    tickers = df.values.flatten().tolist()

    return tickers


if __name__ == '__main__':
    # Read tickers from the CSV
    tickers = read_tickers_from_csv()
    print(f"Total tickers found: {len(tickers)}")
    print("Tickers:", tickers)
    
    # Loop through all tickers and fetch their data
    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        try:
            fetch_and_save_data(ticker)
            print(f"Data fetched successfully for {ticker}!")
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            continue