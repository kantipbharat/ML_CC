import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('out.csv', index_col=0)

def find_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 2 * IQR
    upper_bound = Q3 + 2 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

for column in df.columns:
    outliers = find_outliers(df, column)
    num_outliers = len(outliers)
    print(f'Number of outliers in {column}: {num_outliers}')

    min_val = df[column].min()
    max_val = df[column].max()
    range_val = max_val - min_val
    plot_min = max(min_val - 0.1 * range_val, min(df[column]))
    plot_max = min(max_val + 0.1 * range_val, max(df[column]))

    plt.figure(figsize=(10, 4))
    sns.histplot(df[column], kde=True, bins=30, binrange=(plot_min, plot_max))
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.xlim(plot_min, plot_max)
    plt.show()