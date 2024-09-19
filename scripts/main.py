import csv
with open('../stocks_csv/NVDA.csv', mode ='r')as file:
  csvFile = csv.reader(file)
  for lines in csvFile:
        print(lines)