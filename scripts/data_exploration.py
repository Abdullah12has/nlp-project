import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_path):
    return pd.read_csv(file_path)

def plot_feature_distribution(df, feature):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()

# Example usage:
# df = load_data('data/senti_df.csv')
# plot_feature_distribution(df, 'year')
