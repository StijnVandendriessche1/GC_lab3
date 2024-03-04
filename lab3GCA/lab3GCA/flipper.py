import pandas as pd

# Read the original CSV file
original_df = pd.read_csv('strideTimesBlocksavg32.csv')

# Pivot the data to create a new DataFrame with columns for each amount of blocks
new_df = original_df.pivot(index='amount_of_threads', columns='amount_of_blocks', values='time_to_finish_task')

# Save the new DataFrame to a new CSV file
new_df.to_csv('strideTimesBlocksXSavg_pivoted.csv', index=True)
