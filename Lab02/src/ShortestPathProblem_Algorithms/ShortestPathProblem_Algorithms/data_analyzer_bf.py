import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV data with error handling
csv_file = 'Results/BellmanFord/BellmanFordTests.csv'

#columns = [
#    'node_count',
#    'temperature',
#    'final_edge_count',
#    'runtime_in_ms',
#    'path_was_found',
#    'negative_weights_included'
#]


try:
    #df = pd.read_csv(csv_file, header=None, names=columns)
    df = pd.read_csv(csv_file)
except pd.errors.ParserError as e:
    print(f"Error parsing CSV: {e}")
    exit(1)

# Drop rows with missing values in required columns
required_columns = ['node_count', 'temperature', 'runtime_in_ms', 'path_was_found', 'negative_weights_included']
df = df.dropna(subset=required_columns)

# Ensure columns are numeric (invalid strings will turn into NaN and get dropped)
for col in required_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=required_columns)

# Compute edge count
df['computed_edge_count'] = df['node_count'] * (df['temperature'] + 1)

# Plot
sns.set_theme(style="whitegrid")
for value in [True, False]:
    subset = df[df['negative_weights_included'] == value]
    for found in [True, False]:
        second_subset = subset[df['path_was_found'] == found]
        plt.figure(figsize=(12, 6))
        sns.violinplot(
            x='computed_edge_count',
            y='runtime_in_ms',
            data=second_subset,
            inner='quart',
            scale='width',
            palette='Set2'
        )

        label = 'with' if value else 'without'
        second_label = 'found' if found else 'not found'
        
        if value:
            plt.title(f'Runtime Distribution {label} negative weights. Path {second_label}, weights from -100 to 100')
        else:
            plt.title(f'Runtime Distribution {label} negative weights. Path {second_label} weights from 0 to 100')
                
        plt.xlabel('Computed Edge Count (node_count * (temperature + 1))')
        plt.ylabel('Runtime (ms)')
        #plt.yscale('log')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        sns.pointplot(
        x='computed_edge_count',
        y='runtime_in_ms',
        data=second_subset,
        estimator='mean',
        errorbar='sd',  # Standard deviation
        color='black',
        markers='D',
        linestyles='--'
        )

        label = 'with' if value else 'without'
        second_label = 'found' if found else 'not found'
        
        if value:
            plt.title(f'Runtime Distribution {label} negative weights. Path {second_label}, weights from -100 to 100')
        else:
            plt.title(f'Runtime Distribution {label} negative weights. Path {second_label} weights from 0 to 100')
                
        plt.xlabel('Computed Edge Count (node_count * (temperature + 1))')
        plt.ylabel('Runtime (ms)')
        #plt.yscale('log')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()