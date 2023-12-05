import os
import pandas as pd
import matplotlib.pyplot as plt


def extract_values_from_list(in_list):
    ret_dict = {}
    # Extract the latency numbers
    cur_line = in_list[1] # latency data
    cur_line_split = cur_line.split(' ')
    ret_dict['p50'] = float(cur_line_split[0]) * 1000
    ret_dict['p90'] = float(cur_line_split[1]) * 1000
    ret_dict['p95'] = float(cur_line_split[2]) * 1000

    # Extract Throughput
    cur_line = in_list[3]
    cur_line_split = cur_line.split(' ')
    ret_dict['throughput'] = float(cur_line_split[-1])

    return ret_dict

dir_name = os.getenv('LOG_DIRECTORY')
report_dir_name = os.getenv('REPORT_DIRECTORY', './report')

assert dir_name is not None, 'Specify LOG_DIRECTORY environment variable'

if (dir_name):
    file_list = os.listdir(dir_name)
    # Extract *.log files
    file_list = [x for x in file_list if x.endswith('.log')]

    extract_data_list = []
    for cur_file in file_list:
        with open('%s/%s'%(dir_name, cur_file)) as fp:
            file_contents = fp.readlines()
            extract_data = extract_values_from_list(file_contents)

            # load batch_size and model file name data from name of file
            # file name:  f"model-{model_name}-bs-{batch_size}-{device}.log"
            cur_file_no_ext = cur_file.split('.')[0] # exclued .log
            cur_file_no_ext_split = cur_file_no_ext.split('-')
            extract_data['batch_size'] = int(cur_file_no_ext_split[-2])
            extract_data['device'] = cur_file_no_ext_split[-1]
            extract_data['model_name'] = cur_file_no_ext_split[1]
            extract_data['file_name'] = cur_file_no_ext
            
            extract_data_list.append(extract_data)

    extract_pd = pd.DataFrame.from_dict(extract_data_list)

    extract_pd_sorted_by_batch_size = extract_pd.sort_values(by='batch_size', ascending=True)
    print(extract_pd_sorted_by_batch_size.to_string())

    # Ensure the report directory exists
    os.makedirs(report_dir_name, exist_ok=True)
    model_name = extract_pd['model_name'].iloc[0]
    device = extract_pd['device'].iloc[0]
    DPI = 300
    # Plot Batch-size vs Throughput
    plt.figure(figsize=(8, 6)) 
    plt.plot(extract_pd_sorted_by_batch_size['batch_size'], extract_pd_sorted_by_batch_size['throughput'], marker='o')
    plt.xlabel('Batch Size')
    plt.ylabel('Throughput (QPS)')
    plt.ylim(bottom=0, top=extract_pd['throughput'].max() * 1.1)  
    plt.title(f'{model_name}: Batch Size vs Throughput')
    plt.savefig(f'{report_dir_name}/{model_name}_batch_size_vs_throughput_{device}.png', dpi=DPI)

    # Sort the DataFrame by 'p95' for plotting
    extract_pd_sorted_by_latency = extract_pd.sort_values(by='p95')

    # Plot Latency (P95) vs Throughput
    plt.figure(figsize=(8, 6)) 
    plt.plot(extract_pd_sorted_by_latency['p95'], extract_pd_sorted_by_latency['throughput'], marker='o')
    plt.xlabel('P95 Latency (ms)')
    plt.ylabel('Throughput (QPS)')
    plt.ylim(bottom=0, top=extract_pd['throughput'].max() * 1.1)  
    plt.title(f'{model_name}: Latency vs Throughput')
    plt.savefig(f'{report_dir_name}/{model_name}_latency_vs_throughput_{device}.png', dpi=DPI)

    # Plot Batch Size vs Latency (P95)
    plt.figure(figsize=(8, 6)) 
    plt.plot(extract_pd_sorted_by_batch_size['batch_size'], extract_pd_sorted_by_batch_size['p95'], marker='o')
    plt.xlabel('Batch Size')
    plt.ylabel('P95 Latency (ms)')
    plt.ylim(bottom=0, top=extract_pd['p95'].max() * 1.1)  
    plt.title(f'{model_name}: Batch Size vs Latency (P95)')
    plt.savefig(f'{report_dir_name}/{model_name}_batch_size_vs_latency_{device}.png', dpi=DPI)