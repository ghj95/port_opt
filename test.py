import yfinance as yf

tsx60 = yf.download("TX60.TS", start="2019-01-01", end="2024-01-01")["Close"]           #index
tsx60.rename(columns={"TX60.TS": "TSX 60"}, inplace=True)

normalized_index = tsx60 / tsx60.iloc[0] * 100          # normalize to start at 100

print(tsx60.head())