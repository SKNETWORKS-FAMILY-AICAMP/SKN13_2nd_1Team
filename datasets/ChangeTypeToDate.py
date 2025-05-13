import pandas as pd

# read CSV file.
df = pd.read_csv(r'C:\Workspace\Python\archive\Client Cancellations0.csv')


# Convert datetime format
# df["Cancel Date"] = pd.to_datetime(df["Cancel Date"]).dt.strftime('%Y-%m-%d')
# df["Booking Date"] = pd.to_datetime(df["Booking Date"]).dt.strftime('%Y-%m-%d')


# Convert blank values to null
df['Days'] = pd.to_numeric(df['Days'], errors='coerce')

# 3. NaN 값(결측값)을 0.0으로 대체
df['Days'] = df['Days'].fillna(0.0)


# save the updated DataFrame to a CSV file
df.to_csv(r'C:\Workspace\Python\archive\Client Cancellations0.csv', index=False)

