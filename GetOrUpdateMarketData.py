import requests
import argparse
import csv
import os
from datetime import datetime

def download_historical_data(ticker, multiplier, timespan, api_key, filepath):
    polygon_ticker = f"X:{ticker}"
    base_url = f"https://api.polygon.io/v2/aggs/ticker/{polygon_ticker}/range/{multiplier}/{timespan}/2016-01-09/{datetime.now().strftime('%Y-%m-%d')}?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"
    filename = os.path.join(filepath, f"{ticker.replace(':', '_')}.csv")

    # Check if the file exists and find the last timestamp
    last_timestamp = None
    if os.path.exists(filename):
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header row
            for row in reader:
                last_timestamp = row[0]

    # If the file exists, update the base_url to start from the last timestamp
    if last_timestamp:
        last_timestamp_datetime = datetime.fromtimestamp(int(last_timestamp) // 1000)  # Convert Unix timestamp to datetime
        base_url = f"https://api.polygon.io/v2/aggs/ticker/{polygon_ticker}/range/{multiplier}/{timespan}/{last_timestamp_datetime.strftime('%Y-%m-%d')}/{datetime.now().strftime('%Y-%m-%d')}?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"

    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not last_timestamp:
            writer.writerow(['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP'])

        while base_url:
            try:
                response = requests.get(base_url)
                response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
                data = response.json()

                if data['status'] == 'OK':
                    results = data.get('results', [])
                    for result in results:
                        writer.writerow([
                            result['t'],
                            result['o'],
                            result['h'],
                            result['l'],
                            result['c'],
                            result['v'],
                            result.get('vw')  # Use get() to handle missing 'vw' key
                        ])
                    print(f"Retrieved {len(results)} data points for {ticker}.")
                    next_url = data.get('next_url')
                    if next_url:
                        base_url = f"{next_url}&apiKey={api_key}"
                    else:
                        base_url = None
                else:
                    print(f"Failed to retrieve data for {ticker}. Status: {data.get('status')}")
                    print(f"Error message: {data.get('error')}")
                    break
            except requests.exceptions.RequestException as e:
                print(f"Error occurred while retrieving data for {ticker}: {e}")
                break
            except (KeyError, ValueError) as e:
                print(f"Error occurred while processing data for {ticker}: {e}")
                break

    print(f"Data saved to {filename}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download historical aggregate data for currency tickers.')
    parser.add_argument('ticker', type=str, nargs='?', help='The currency ticker symbol (e.g., BTCUSD)')
    parser.add_argument('-m', '--multiplier', type=int, default=1, help='The size of the timespan multiplier (default: 1)')
    parser.add_argument('-t', '--timespan', type=str, default='minute', choices=['minute', 'hour', 'day'], help='The size of the time window (default: minute)')
    apikey = os.getenv('POLYGON_API_KEY')

    if apikey is None:
        raise ValueError("API key not found. Please set the POLYGON_API_KEY environment variable.")

    filepath = './Data'
    if not os.path.exists(filepath):
        os.makedirs(filepath)
        
    args = parser.parse_args()
    
    if args.ticker:
        download_historical_data(args.ticker, args.multiplier, args.timespan, apikey, filepath)
    else:
        with open('tickers.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                for ticker in row:
                    ticker = ticker.strip()
                    if ticker:
                        download_historical_data(ticker, args.multiplier, args.timespan, apikey, filepath)