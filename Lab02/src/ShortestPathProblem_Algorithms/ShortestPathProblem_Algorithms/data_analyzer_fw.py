import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV data with error handling
csv_file = 'Results/FloydWarshall/FloydWarshallTests.csv'

#columns = [
#    'node_count',
#    'temperature',
#    'final_edge_count',
#    'runtime_in_ms',
#    'negative_cycle_not_detected',
#    'negative_weights_included', 
#    'weight'
#]


try:
    #df = pd.read_csv(csv_file, header=None, names=columns)
    df = pd.read_csv(csv_file)
except pd.errors.ParserError as e:
    print(f"Error parsing CSV: {e}")
    exit(1)

# Drop rows with missing values in required columns
required_columns = ['node_count', 'temperature', 'runtime_in_ms', 'negative_cycle_not_detected', 'negative_weights_included', 'weight']
df = df.dropna(subset=required_columns)

# Ensure columns are numeric (invalid strings will turn into NaN and get dropped)
for col in required_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=required_columns)

# Compute edge count
df['computed_edge_count'] = df['node_count'] * (df['temperature'] + 1)

# Plot
sns.set_theme(style="whitegrid")
for negW in [True, False]:
    subset = df[df['negative_weights_included'] == negW]
    for not_detected in [True, False]:
        second_subset = subset[df['negative_cycle_not_detected'] == not_detected]
        for value in [30, 50, 100]:
            # this should be on top, then neg_weights -> then neg_cycle
            third_subset = second_subset[df['weight'] == value]

            plt.figure(figsize=(12, 6))
            sns.violinplot(
                x='computed_edge_count',
                y='runtime_in_ms',
                data=third_subset,
                inner='quart',
                scale='width',
                palette='Set2'
            )

            label = 'with' if negW else 'without'
            second_label = 'not detected' if not_detected else 'detected'
            
            if negW:
                plt.title(f'Runtime Distribution {label} Negative Weights negative cycle {second_label}, weights from {-value} to {value}')
            else:
                plt.title(f'Runtime Distribution {label} Negative Weights negative cycle {second_label} weights from 0 to {value}')
                    
            plt.xlabel('Computed Edge Count (node_count * (temperature + 1))')
            plt.ylabel('Runtime (ms)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()