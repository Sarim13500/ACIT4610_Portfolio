import pandas as pd
import os


def calculate_monthly_returns(stock_file, stock_folder, results_folder):
    """Calculate and save the monthly returns for each stock."""
    # Read the stock CSV file
    file_path = os.path.join(stock_folder, stock_file)
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)

    # Calculate monthly returns
    df_monthly = df['Close'].resample('M').ffill().pct_change()

    # Save the monthly returns to the results folder
    result_file_path = os.path.join(results_folder, stock_file)
    df_monthly.to_csv(result_file_path)
    print(f"Monthly returns saved to {result_file_path}")


def process_all_stocks(stock_folder, results_folder):
    """Process all stock files in the stocks_csv folder."""
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    stock_files = [f for f in os.listdir(stock_folder) if f.endswith('.csv')]
    for stock_file in stock_files:
        calculate_monthly_returns(stock_file, stock_folder, results_folder)


if __name__ == "__main__":
    stock_folder = "../stocks_csv"
    results_folder = "../results/stocks_results"

    process_all_stocks(stock_folder, results_folder)
