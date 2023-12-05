
import os
import random
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

from tqdm import tqdm
from common_settings import default_image_size, default_batch_size

image_size = int(os.getenv('IMAGE_SIZE', default_image_size))
batch_size = int(os.getenv('BATCH_SIZE', default_batch_size))

# Benchmark test parameters
num_models = 2  # num_models <= number of cores (4 for inf1.xl and inf1.2xl, 16 for inf1.6xl)
num_threads = 2  # Setting num_threads to num_models works well.
num_requests = 100
num_request_samples = 10
half_precision = True
data_dir = './data'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

ts_model_file = 'resnet50_%s_%d_%d.pt'%(device, image_size, batch_size)
model_name = 'resnet50'

preprocess = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])

print(f'Model: {model_name}')
print('Image Size: %d, Batch Size: %d, Half Precision: %r'%(image_size, batch_size, half_precision))

# Create a pipeline with the given model
# model_dict = dict()
# model_dict['return_dict'] = False

# Load Images from the Folder
img_preprocessed_list = []
jpg_file_list = os.listdir(data_dir)
jpg_file_list = [x for x in jpg_file_list if '.jpg' in x]
jpg_file_list_sample = random.sample(jpg_file_list, num_request_samples)

for cur_image_file in jpg_file_list_sample:
    cur_image = Image.open('%s/%s' % (data_dir, cur_image_file)).convert('RGB')

    cur_image_preprocessed = preprocess(cur_image)
    cur_image_preprocessed_unsqueeze = torch.unsqueeze(cur_image_preprocessed, 0)
    img_preprocessed_list.append(cur_image_preprocessed_unsqueeze)


def load_model(file_name, torchscript):
    # Load modelbase
    with torch.cuda.amp.autocast(enabled=half_precision):
        if torchscript:
            model = torch.jit.load(file_name)
            model.eval()
            model = model.to(device)
        else:
            model = resnet50(weights=ResNet50_Weights.DEFAULT) # TODO: change model in a parameter in future?
            model.eval()
            model = model.to(device)

    return model

latency_list = []

def task(model, cur_img_preprocess):
    global latency_list
    begin = time.time()
    with torch.cuda.amp.autocast(enabled=half_precision):
        batch_input_tensor = torch.cat([cur_img_preprocess] * batch_size).to(device)
        prediction = model(batch_input_tensor)
        latency_time = time.time() - begin
        latency_list.append(latency_time)
    return


def benchmark(num_models, num_threads, num_requests, model_file, torchscript=True):
    # Load a model into each NeuronCore
    print('Loading Models To Memory')
    models = [load_model(model_file, torchscript) for _ in range(num_models)]
    print('Starting benchmark')
    output_list = []
    begin = time.time()
    futures = []
    # Submit all tasks and wait for them to finish
    # https://stackoverflow.com/questions/51601756/use-tqdm-with-concurrent-futures
    with tqdm(total=num_requests) as pbar:
        with ThreadPoolExecutor(num_threads) as pool:
            for i in range(num_requests):
                futures.append(pool.submit(task, models[i % len(models)], random.choice(img_preprocessed_list)))
                #output_list.append(output.result())
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)

    test_time = time.time() - begin
    return test_time


# test_time, latency_array = benchmark(num_models, num_threads, num_requests, model_file, torchscript=True)
test_time = benchmark(num_models, num_threads, num_requests, ts_model_file, torchscript=False)
latency_data = np.percentile(np.array(latency_list), [50, 90, 95])

print('Latency: (P50, P90, P95)')
print(latency_data)
print('Total time taken for %d * (%d images) is %0.4f seconds' % (num_requests, batch_size, test_time))
print('Throughput (num_requests * batch_size /sec) = %0.4f' % (num_requests * batch_size / test_time))

dir_name = os.getenv('LOG_DIRECTORY', 'default_log_dir')
os.makedirs(dir_name, exist_ok=True)
log_filename = f"{dir_name}/model-{model_name}-bs-{batch_size}-{device}.log"

with open(log_filename, 'w') as file:
    file.write(f"Latency: (P50, P90, P95)\n")
    file.write(f"{latency_data[0]} {latency_data[1]} {latency_data[2]}\n")
    file.write(f"Total time taken for {num_requests} * ({batch_size} images) is {test_time:.4f} seconds\n")
    file.write(f"Throughput (num_requests * batch_size /sec) = {(num_requests * batch_size / test_time):.4f}\n")

