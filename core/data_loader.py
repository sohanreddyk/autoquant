import yfinance as yf
import pandas as pd

def download_data(tickers, start, end, save_path="data/stock_data.csv"):
    df = yf.download(tickers, start=start, end=end, group_by='ticker')
    df.to_csv(save_path)
    print(f"✅ Data saved to {save_path}")
    return df

# Example usage
if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL"]
    download_data(tickers, "2018-01-01", "2023-12-31")
