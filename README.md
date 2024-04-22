# Cryptocurrency Market Analysis and Prediction

This project aims to analyze and predict cryptocurrency market trends using market data, sentiment analysis, and news events. It retrieves data from external APIs, processes the data, and creates datasets that can be used for visualization and training machine learning models.

## Table of Contents

- [Getting Started](#getting-started)
 - [Prerequisites](#prerequisites)
 - [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Getting Started

These instructions will help you set up the project on your local machine for development and testing purposes.

### Prerequisites

To run this project, you need to have the following:

- Python 3.x
- API token from [Polygon.io](https://polygon.io/)
- API token from [CryptoNews API](https://cryptonews-api.com/)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/cryptocurrency-market-analysis.git
```

2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

3. Set the environment variables for API tokens:

```bash
export POLYGON_API_KEY=your_polygon_api_key
export CRYPTO_NEWS_API_TOKEN=your_cryptonews_api_token
```

## Usage

1. Retrieve market data from Polygon.io by running:

```bash
python GetOrUpdateMarketData.py
```

2. Retrieve sentiment data from CryptoNews API by running:

```bash
python RetrieveSentiment.py
```

3. Retrieve news events from CryptoNews API by running:

```bash
python RetrieveNewsEvents.py
```
4. Create datasets for visualization and training machine learning models by running:

```bash
python DatasetCreator.py
```

## Project Structure

The project consists of the following modules:

- `GetOrUpdateMarketData.py`: Retrieves market data from Polygon.io.
- `RetrieveSentiment.py`: Retrieves sentiment data from CryptoNews API.
- `RetrieveNewsEvents.py`: Retrieves news events from CryptoNews API.
- `DatasetCreator.py`: Creates datasets based on market information for visualization and training machine learning models.
- `TaCalcs.py`: A library of technical analysis functions used by the dataset creation program.
- `tickers.csv`: A list of cryptocurrency tickers. Used as the default input for the MarketData and DatasetCreator modules.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
