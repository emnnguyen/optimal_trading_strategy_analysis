# libraries
import yfinance as yf
import pandas as pd
import time
import os

# configuration
start = "2000-01-01"
end = "2024-12-31"
directory = "."

# selected stocks
stocks = [# S&P 500
          '^GSPC', 
          # technology
          'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'AVGO', 'CSCO', 'ADBE', 'CRM', 'INTC',
          # financials
          'JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'AXP', 'SCHW',
          # healthcare
          'UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'PFE', 'TMO',
          # consumer discretionary
          'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX',
          # consumer staples
          'WMT', 'PG', 'KO', 'PEP', 
          # industrials
          'BA', 'CAT', 'UPS', 'RTX', 'HON', 
          # energy
          'XOM', 'CVX', 'COP', 
          # communication services
          'DIS', 'NFLX', 'CMCSA',
          # utilities
          'NEE', 'DUK',
          # real estate 
          'AMT', 'PLD']

# track results
successful = []
failed = []
total = len(stocks)

# download stock data
for i, stock in enumerate(stocks, 1):
    print(f"[{i}/{total}] downloading {stock}")

    try:
        data = yf.download(stock, start=start, end=end, auto_adjust=True)

        # check if empty
        if data.empty:
            print(f"no data")
            failed.append(stock)
            continue
            
        # referenced https://www.geeksforgeeks.org/python/python-pandas-timedeltaindex-get_level_values/
        data.columns = data.columns.get_level_values(0)

        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    
        filename = f"{directory}/{stock}.csv"
        data.to_csv(filename, date_format='%Y-%m-%d')
        
        print(f"saved {len(data)} rows")
        successful.append(stock)

    # referenced https://docs.python.org/3/tutorial/errors.html
    except Exception as err:
        print(f"error: {err}")
        failed.append(stock)
        
    # delay to avoid overloading
    time.sleep(0.5)

# check how many successful or failed
print(f"\nsuccessful: {len(successful)}/{total}")
print(f"failed: {len(failed)}/{total}")

# check for missing values
# referenced https://community.fabric.microsoft.com/t5/Desktop/Bring-in-Multiple-CSVs-with-different-information-Build-up/td-p/2975377
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

missing = []

for file in csv_files:
    filepath = os.path.join(directory, file)
    df = pd.read_csv(filepath)
    
    rows = len(df)
    counts = df.isnull().sum()
    total = counts.sum()
    
    if total > 0:
        missing.append({
            'File': file,
            'Total Rows': rows,
            'Missing Values': total,
            'Missing %': f"{(total / (rows * len(df.columns))) * 100:.2f}%"
        })
        
        print(f"\n{file}:")
        print(f"Total rows: {rows}")
        for col in counts[counts > 0].index:
            print(f"{col}: {counts[col]} missing ({counts[col]/rows*100:.2f}%)")

if missing:
    summary = pd.DataFrame(missing)
    print(summary.to_string(index=False))
else:
    print("\nNo missing values found in any files")

print(f"\nTotal files checked: {len(csv_files)}")