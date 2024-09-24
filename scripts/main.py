import pandas as pd
import os

# Use the absolute path to your stocks_csv directory
stocks_path = "/Users/sarim/Desktop/@ACIT4610/Portfolio/stocks_csv"
file_path = os.path.join(stocks_path, "AAPL.csv")

# Check if the file exists
if not os.path.isfile(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist.")

# Load the CSV file
df = pd.read_csv(file_path)

# Print the column names to see what they are
print(df.columns)
