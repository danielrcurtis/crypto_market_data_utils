import requests
import csv
import os
from datetime import datetime, timedelta

def fetch_sentiment_data(ticker, date_range, token):
    url = f"https://cryptonews-api.com/api/v1/stat?&tickers={ticker}&date={date_range}&token={token}"
    response = requests.get(url)
    data = response.json()
    return data

def write_to_csv(file_path, data, header, ticker):
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)
        for date, sentiment in data.items():
            if not any(date in row for row in csv.reader(open(file_path))):
                writer.writerow([date, sentiment[ticker]['sentiment_score']])

def get_last_update(file_path):
    if not os.path.isfile(file_path):
        return None
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        rows = list(reader)
        if rows:
            return rows[-1][0]  # Return the last date in the CSV
    return None

def main():
    token = os.getenv('CRYPTO_NEWS_API_TOKEN')
    if token is None:
        raise ValueError("API token not found. Please set the CRYPTO_NEWS_API_TOKEN environment variable.")
        
    tickers_file = 'tickers.csv'
    default_range = '01152016-today'
    sentiment_folder = './Sentiment/'
    if not os.path.exists(sentiment_folder):
        os.makedirs(sentiment_folder)

    with open(tickers_file, 'r') as file:
        tickers = [ticker.strip().replace('USD', '') for ticker in file.read().split(',')]

    for ticker in tickers:
        file_path = os.path.join(sentiment_folder, f"{ticker}_sentiment.csv") #file_path = f"{ticker}_sentiment.csv"  
        last_update = get_last_update(file_path)

        if last_update:
            last_update_date = datetime.strptime(last_update, '%Y-%m-%d')
            today = datetime.now().date()
            days_since_update = (today - last_update_date.date()).days
            date_range = f"last{days_since_update}days"
        else:
            date_range = default_range

        data = fetch_sentiment_data(ticker, date_range, token)

        sentiment_data = data['data']

        write_to_csv(file_path, sentiment_data, ['Date', 'Sentiment_Score'], ticker)

if __name__ == '__main__':
    main()