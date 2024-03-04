import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming your dataset is in a CSV file named 'gpu_data.csv'
# Load the data into a pandas DataFrame
gpu_data = pd.read_csv('strideTimesBlocksXXSavg2.csv')

# Visualizing the amount of blocks used vs time
sns.scatterplot(data=gpu_data, x='amount_of_blocks', y='time_to_finish_task')
plt.xlabel('Amount of Blocks Used')
plt.ylabel('Time to Finish Task (seconds)')
plt.title('Amount of Blocks Used vs Time to Finish Task')
plt.savefig('amount_of_blocks_vs_timexs.png')
plt.show()

# Visualizing the amount of threads used vs time
sns.scatterplot(data=gpu_data, x='amount_of_threads', y='time_to_finish_task')
plt.xlabel('Amount of Threads Used')
plt.ylabel('Time to Finish Task (seconds)')
plt.title('Amount of Threads Used vs Time to Finish Task')
plt.savefig('amount_of_threads_vs_timexs.png')
plt.show()

# Visualizing the amount of blocks and threads used vs time
sns.scatterplot(data=gpu_data, x='amount_of_blocks', y='amount_of_threads', hue='time_to_finish_task')
plt.xlabel('Amount of Blocks Used')
plt.ylabel('Amount of Threads Used')
plt.title('Amount of Blocks and Threads Used vs Time to Finish Task')
plt.savefig('amount_of_blocks_and_threads_vs_timexs.png')
plt.show()
