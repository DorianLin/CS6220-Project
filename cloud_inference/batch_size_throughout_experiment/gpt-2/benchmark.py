import os
import random
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from essential_generators import DocumentGenerator
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from common_settings import default_model_name, default_batch_size, default_max_length


# Environment variables and constants
max_length = int(os.getenv('MAX_LENGTH', default_max_length))
batch_size = int(os.getenv('BATCH_SIZE', default_batch_size))
model_name = os.getenv('MODEL_NAME', default_model_name)

# Benchmark test parameters
num_models = 1
num_threads = 1
num_requests = 100
num_request_samples = 10
half_precision = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

ts_model_file = '%s_%s_%d.pt' % (model_name, device ,max_length)
print(f'Max Length: {max_length}, Batch Size: {batch_size}, Model Name: {model_name}, Half Precision: {half_precision}')

# Create a pipeline with the given model
model_dict = dict()
model_dict['return_dict'] = False
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Generate document samples
gen = DocumentGenerator()
sequence_list = []
encoded_input_list = []
for _ in np.arange(num_request_samples):
    sequence = gen.sentence()
    encoded_inputs = tokenizer.encode(sequence, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
    sequence_list.append(sequence)
    encoded_input_list.append(encoded_inputs)

def load_model(file_name, torchscript):
    with torch.cuda.amp.autocast(enabled=half_precision):
        if torchscript:
            model = torch.jit.load(file_name)
        else:
            model = GPT2LMHeadModel.from_pretrained(model_name)

        model.eval()
        model = model.to(device)

    return model

latency_list = []

def task(model, encoded_inputs):
    global latency_list
    begin = time.time()


    with torch.cuda.amp.autocast(enabled=half_precision):
        input_ids_tensor = encoded_inputs['input_ids']
        batch_input_ids_tensor = torch.cat([input_ids_tensor] * batch_size)
        # attention_mask_tensor = encoded_inputs['attention_mask']
        # batch_attention_mask_tensor = torch.cat([attention_mask_tensor] * batch_size)
        # ts_input = batch_input_ids_tensor.to(device), batch_attention_mask_tensor.to(device)
        # neuron_input = encoded_input['input_ids'], encoded_input['attention_mask']
        _ = model(batch_input_ids_tensor, max_length=max_length, num_return_sequences=1)
        latency_time = time.time() - begin

        latency_list.append(latency_time)
    return




def benchmark(num_models, num_threads, num_requests, model_file, torchscript=True):
    # Load a model into each NeuronCore
    print('Loading Models To Memory')
    models = [load_model(model_file, torchscript) for _ in range(num_models)]
    tokenizers = [tokenizer for _ in range(num_models)]
    print('Starting benchmark')
    output_list = []
    begin = time.time()
    futures = []
    # Submit all tasks and wait for them to finish
    # https://stackoverflow.com/questions/51601756/use-tqdm-with-concurrent-futures
    with tqdm(total=num_requests) as pbar:
        with ThreadPoolExecutor(num_threads) as pool:
            for i in range(num_requests):
                # futures.append(pool.submit(task, models[i % len(models)], tokenizers[i % len(models)], random.choice(sequence_list)))
                futures.append(pool.submit(task, models[i % len(models)], random.choice(encoded_input_list)))
                # output_list.append(output.result())
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)

    test_time = time.time() - begin

    # return test_time, np.array(output_list)
    return test_time
    

# Run the benchmark and log and print results
test_time = benchmark(num_models, num_threads, num_requests, ts_model_file, torchscript=True)
latency_data = np.percentile(np.array(latency_list), [50, 90, 95])

# Printing results
print(f'Latency: (P50, P90, P95)')
print(latency_data)
print(f'Total samples: {len(latency_list)}')
print('Total time taken for %d * (%d sentences) is %0.4f seconds' % (num_requests, batch_size, test_time))
print('Throughput (num_requests * batch_size /sec) = %0.4f' % (num_requests * batch_size / test_time))


dir_name = os.getenv('LOG_DIRECTORY', 'default_log_dir')
os.makedirs(dir_name, exist_ok=True)
log_filename = f"{dir_name}/model-{model_name}-bs-{batch_size}-{device}.log"

with open(log_filename, 'w') as file:
    file.write(f"Latency: (P50, P90, P95)\n")
    file.write(f"{latency_data[0]} {latency_data[1]} {latency_data[2]}\n")
    file.write(f"Total time taken for {num_requests} * ({batch_size} images) is {test_time:.4f} seconds\n")
    file.write(f"Throughput (num_requests * batch_size /sec) = {(num_requests * batch_size / test_time):.4f}\n")

