import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
from datetime import datetime
from collections import defaultdict
import numpy as np

def read_log_file(file_paths):
    
    datas = defaultdict(lambda: defaultdict(lambda: dict()))
    for file_path in file_paths:
        initial_timestamp = None
        with open(file_path, 'r') as file:
            for line_number, line in enumerate(file, start=1):
                try:
                    if ':ADKPlot:' not in line:
                        continue  # Skip lines without a colon, as they likely don't contain JSON
                    timestamp, json_str = line.strip().split(':ADKPlot:', 1)
                    timestamp = int(timestamp)
                    if(initial_timestamp is None):
                        initial_timestamp = timestamp
                    # Attempt to decode JSON, skip lines with invalid JSON
                    try:
                        record = json.loads(json_str)
                    except json.JSONDecodeError:
                        print(f"Skipping invalid JSON at line {line_number}")
                        continue
                    
                    dt_timestamp = timestamp-initial_timestamp
                    for k, vs in record.items():
                        vs = np.array(vs).astype(float)
                        if(len(vs.shape)==1):
                            vs = np.expand_dims(vs, axis=0)
                 
                        means = np.mean(np.array(vs), axis=0)
                        stds = np.std(np.array(vs), axis=0)
                        for i in range(means.shape[0]):
                            datas[f"{k}[{i}]"][dt_timestamp]["mean"] = means[i]
                            datas[f"{k}[{i}]"][dt_timestamp]["std"] = stds[i]
                  
                        
                except ValueError as e:
                    print(f"Error processing line {line_number}: {e}")
                    continue

    return datas

def plot_data(datas):
    plt.figure(figsize=(12, 6))
    
    print(datas.keys())
    # sim = "isaac"
    sim = "araas"
    # Z end effector force
    # keys = [[f"{sim}_action[{i}]" for i in range(3)]]
    keys = [[f"{sim}_obs[{i}]" for i in range(3)]]
    # keys += [[f"{sim}_obs[{i}]" for i in range(3, 9)]]
    sim = "isaac"
    # Z end effector force
    # keys += [[f"robot1_{sim}_action[{i}]" for i in range(3)]]
    keys += [[f"robot1_{sim}_obs[{i}]" for i in range(3)]]
    # keys += [[f"robot1_{sim}_obs[{i}]" for i in range(3, 9)]]
    # keys = [["Robot1_target_velocity[0]", "Robot1_target_velocity[1]", "Robot1_target_velocity[2]"]]
    start_time = 0
    for ki, key_group in enumerate(keys):
        # Plot joint forces
        ax = plt.subplot(1, len(keys), ki+1)
        plt.title('ADK Plot')
        plt.xlabel('Time')
        plt.ylabel('Value')
        
        legend_lines = []
        legend_titles = []
        for key in key_group:
            timestamps = list(datas[key].keys())
            means = []
            stds = []
            visible_timestamps = []
            for timestamp in timestamps:
                if(timestamp>start_time):
                    means.append(datas[key][timestamp]["mean"])
                    stds.append(datas[key][timestamp]["std"])
                    visible_timestamps.append(timestamp)

            line, = ax.plot(visible_timestamps, means, label='Joint Forces First Element')
            legend_lines.append(line)
            legend_titles.append(key)
            std_upper = [m + s for m, s in zip(means, stds)]
            std_lower = [m - s for m, s in zip(means, stds)]
            ax.fill_between(visible_timestamps, std_lower, std_upper, alpha=0.15, label=f'{key} Std Dev')

        ax.legend(legend_lines, legend_titles)
      

    plt.savefig('log_plot.pdf', format='pdf')
    plt.tight_layout()
    plt.show()

def main():
    # log_file_paths = ['/home/aidanc/multi-robot-assembly/multi_arm_assembly/logs/1719536935.3326485.log']
    log_file_paths=['/home/aidanc/multi-robot-assembly/multi_arm_assembly/logs/1719628939.3054402.log', 
                    '/home/aidanc/multi-robot-assembly/multi_arm_assembly/logs/1719628979.538773.log']
    datas = read_log_file(log_file_paths)
    plot_data(datas)

if __name__ == '__main__':
    main()