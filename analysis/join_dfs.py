import pandas as pd

# join the two dataframes on id

df1 = pd.read_parquet("generation_results_draftvlm.parquet")
df2 = pd.read_parquet("instructions_with_category_subcategory.parquet")

# Reset index to make the row index a column
df1 = df1.reset_index()
df2 = df2.reset_index()

print(df1.head(3))
print("==========================")
print(df2.head(3))


# Merge the dataframes on the new 'index' column
merged_df = pd.merge(df1, df2, on='index', how='inner')

# Save the merged dataframe to a new parquet file
merged_df.to_parquet("merged_results.parquet")