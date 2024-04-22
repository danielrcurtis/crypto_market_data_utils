import requests
import csv
import os
from datetime import datetime
import time

def fetch_event_data(ticker, page, token):
    url = f"https://cryptonews-api.com/api/v1/events?&tickers={ticker}&page={page}&token={token}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while fetching data: {e}")
        return None
    except ValueError as e:
        print(f"Error occurred while parsing JSON: {e}")
        return None

def write_to_csv(file_path, data, header):
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)
        for event in data:
            if not any(event['event_id'] in row for row in csv.reader(open(file_path, encoding='utf-8'))):
                writer.writerow([event['event_id'], event['event_name'], event['event_text'], event['date']])

def get_last_event_date(file_path):
    if not os.path.isfile(file_path):
        return None
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        rows = list(reader)
        if rows:
            return rows[-1][3]  # Return the date of the last event in the CSV
    return None

def main():
    token = os.getenv('CRYPTO_NEWS_API_TOKEN')
    if token is None:
        raise ValueError("API token not found. Please set the CRYPTO_NEWS_API_TOKEN environment variable.")
        
    tickers_file = 'tickers.csv'
    events_folder = './NewsEvents/'
    if not os.path.exists(events_folder):
        os.makedirs(events_folder)

    with open(tickers_file, 'r') as file:
        tickers = [ticker.strip().replace('USD', '') for ticker in file.read().split(',')]

    for ticker in tickers:
        file_path = os.path.join(events_folder, f"{ticker}_events.csv")
        last_event_date = get_last_event_date(file_path)

        page = 1
        while True:
            data = fetch_event_data(ticker, page, token)
            if data is None:
                break

            events = data['data']
            if not events:
                break

            write_to_csv(file_path, events, ['event_id', 'event_name', 'event_text', 'date'])

            if last_event_date:
                event_dates = [event['date'] for event in events]
                if last_event_date in event_dates:
                    break

            page += 1
            time.sleep(1)  # Add a 1-second delay between requests

if __name__ == '__main__':
    main()