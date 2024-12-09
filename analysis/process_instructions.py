import pandas as pd

# Read CSV file directly into DataFrame
df = pd.read_csv('instructions_500.csv', header=None)

# Transpose and convert to DataFrame
df = df.T
df.columns = ['instruction']

# Save DataFrame to CSV file with index
df.to_parquet('instructions_with_id.parquet')

# print head
print(df.head(4))