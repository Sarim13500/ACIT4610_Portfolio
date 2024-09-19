import yfinance as yf
import csv
import os

# Define the ticker symbol and the date range
ticker = "COST"
start = "2018-01-01"
end = "2022-12-31"

# Download stock data
stock_data = yf.download(ticker, start=start, end=end)

# Define the file path (relative to the script location)
file_path = os.path.join(os.path.dirname(__file__), "../stocks_csv", f"{ticker}.csv")

# Ensure the directory exists
directory = os.path.dirname(file_path)
if not os.path.exists(directory):
    os.makedirs(directory)

# Write the data to the CSV file
try:
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the header (column names)
        writer.writerow(['Date'] + list(stock_data.columns))

        # Write the data rows with formatted values
        for index, row in stock_data.iterrows():
            formatted_row = [
                f"{val:.6f}" if col != 'Volume' and isinstance(val, (int, float))
                else int(val) if col == 'Volume'
                else val
                for col, val in zip(stock_data.columns, row)
            ]
            writer.writerow([index] + formatted_row)

    print(f"Data saved to {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")
