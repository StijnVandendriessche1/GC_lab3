import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming your dataset is in a CSV file named 'gpu_data.csv'
# Load the data into a pandas DataFrame
gpu_data = pd.read_csv('strideTimesavg32.csv')

# Group the data by amount of blocks used
block_groups = gpu_data.groupby('amount_of_blocks')

# Create a line graph for each amount of blocks used
for block, block_group in block_groups:
    plt.plot(block_group['amount_of_threads'], block_group['time_to_finish_task'], label=f'{block} Blocks')
    plt.xlabel('Amount of Threads Used')
    plt.ylabel('Time to Finish Task (seconds)')
    plt.title('Time to Finish Task as a Function of Threads')
    plt.legend()
    plt.savefig(f'large_time_vs_threads_blocks_{block}.png')
    plt.show()
