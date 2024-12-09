import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, kruskal

# calculate the proportion of 0 and 2 out of total 

df = pd.read_parquet("merged_results.parquet")

print(df.columns)

print(df['instruction_x'])
print(df['instruction_y'])

df = df.drop(columns=['index', 'instruction_y'])
df = df.rename(columns={'instruction_x': 'instruction'})

def get_prop_draft(trajectory_list): 
    nums = np.asarray(trajectory_list)
    count_0_and_2 = np.count_nonzero((nums == 0) | (nums == 2))
    proportion = count_0_and_2 / nums.size
    return proportion

# Apply the function using Pandas apply() method instead of PySpark UDF
df['prop_draft'] = df['trajectory'].apply(get_prop_draft)

print(df["prop_draft"])

# Handle categories
df['category'] = df['category'].apply(eval)  # Convert string repr to list
df = df.explode('category')
df['category'] = df['category'].fillna('NONE')
df.loc[df['category'].apply(lambda x: x == []), 'category'] = 'NONE'

# Calculate stats for categories
category_stats = df.groupby('category').agg({
    'prop_draft': ['mean', lambda x: x.std() / np.sqrt(len(x)), 'count']
}).round(4)

category_stats.columns = ['mean', 'stderr', 'count']
category_stats = category_stats.reset_index()

print("\nStatistics by category:")
print(category_stats)

# Handle subcategories - similar process
df['subcategory'] = df['subcategory'].apply(eval)  # Convert string repr to list
df = df.explode('subcategory')
df['subcategory'] = df['subcategory'].fillna('NONE')
df.loc[df['subcategory'].apply(lambda x: x == []), 'subcategory'] = 'NONE'

# Calculate stats for subcategories
subcategory_stats = df.groupby('subcategory').agg({
    'prop_draft': ['mean', lambda x: x.std() / np.sqrt(len(x)), 'count']
}).round(4)

subcategory_stats.columns = ['mean', 'stderr', 'count']
subcategory_stats = subcategory_stats.reset_index()

print("\nStatistics by subcategory:")
print(subcategory_stats)

# 1. First, get valid subcategories (n≥9, excluding NONE)
min_samples = 9
valid_subcategories = subcategory_stats[
    (subcategory_stats['count'] >= min_samples) & 
    (subcategory_stats['subcategory'] != 'NONE')
]['subcategory'].tolist()

# 2. Create filtered dataframe for subcategories
df_subcat_filtered = df[df['subcategory'].isin(valid_subcategories)].copy()

# 3. Get the parent category by splitting on "-" and taking first element
df_subcat_filtered['parent_category'] = df_subcat_filtered['subcategory'].apply(lambda x: x.split('-')[0])

# 4. Sort categories by mean prop_draft for consistent ordering
parent_means = df_subcat_filtered.groupby('parent_category')['prop_draft'].mean().sort_values()
parent_order = parent_means.index.tolist()

# 5. Create ordered list of subcategories within each parent
subcategory_order = []
for parent in parent_order:
    # Get subcategories for this parent, sorted by their mean prop_draft
    mask = df_subcat_filtered['parent_category'] == parent
    subcats = df_subcat_filtered[mask].groupby('subcategory')['prop_draft'].mean().sort_values().index
    subcategory_order.extend(subcats)

# 6. Create the plot with explicit ordering
plt.figure(figsize=(15, 6))
sns.boxplot(data=df_subcat_filtered, 
            x='subcategory', 
            y='prop_draft', 
            hue='parent_category',
            order=subcategory_order,
            hue_order=parent_order)

plt.xticks(rotation=45, ha='right')
plt.title('Distribution of Draft Proportions by Subcategory (n≥9)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Parent Category')
plt.tight_layout()
plt.savefig('subcategory_boxplots.png', bbox_inches='tight')
plt.close()

# Statistical Tests
print("\nPerforming Kruskal-Wallis H-test and Dunn's post-hoc test...")

# 1. Kruskal-Wallis H-test for categories
category_groups = [df[df['category'] == cat]['prop_draft'].values 
                  for cat in category_stats['category']]
h_stat, p_value = kruskal(*category_groups)

print(f"\nKruskal-Wallis H-test results for categories:")
print(f"H-statistic: {h_stat:.4f}")
print(f"p-value: {p_value:.4e}")

# 2. Dunn's post-hoc test for categories
from scikit_posthocs import posthoc_dunn

# Perform Dunn's test
dunn_results = posthoc_dunn(df, val_col='prop_draft', group_col='category', p_adjust='bonferroni')
print("\nDunn's post-hoc test results for categories (p-values):")
print(dunn_results)

# 3. Repeat tests for valid subcategories (n≥9)
print("\nPerforming tests for subcategories with n≥9...")

subcategory_groups = [df[df['subcategory'] == subcat]['prop_draft'].values 
                     for subcat in valid_subcategories]
h_stat_sub, p_value_sub = kruskal(*subcategory_groups)

print(f"\nKruskal-Wallis H-test results for subcategories:")
print(f"H-statistic: {h_stat_sub:.4f}")
print(f"p-value: {p_value_sub:.4e}")

# Dunn's test for subcategories
dunn_results_sub = posthoc_dunn(df_subcat_filtered, 
                               val_col='prop_draft', 
                               group_col='subcategory', 
                               p_adjust='bonferroni')
print("\nDunn's post-hoc test results for subcategories (p-values):")
print(dunn_results_sub)