import pandas as pd
import os

def calculate_covariance_matrix():
    stock_folder = "../stocks_csv"
    stock_files = [f for f in os.listdir(stock_folder) if f.endswith('.csv')]

    returns = []
    tickers = []
    for file in stock_files:
        ticker = file.replace(".csv", "")
        file_path = f"../results/{ticker}_monthly_returns.csv"
        df = pd.read_csv(file_path, index_col='Date')
        returns.append(df)
        tickers.append(ticker)

    returns_df = pd.concat(returns, axis=1)
    returns_df.columns = tickers
    covariance_matrix = returns_df.cov()

    covariance_path = os.path.join(os.path.dirname(__file__), "../results/returns_matrix.csv")
    covariance_matrix.to_csv(covariance_path)
    print(f"Covariance matrix saved to {covariance_path}")

if __name__ == "__main__":
    calculate_covariance_matrix()
