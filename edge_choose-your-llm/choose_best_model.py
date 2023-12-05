# import packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import get_model_memory
import get_resources_info
from Scraper import scrape
import subprocess
import os

'''
use get_resources_info.py to get if gpu and memory
use scrape.py to get update the leaderboard and return sorted model list
use available memory to find the best model in range
deploy
'''

# get_resources_info.py
print(f'{"="*110}\nMeasuring Local Resources...\n')
local_resources = get_resources_info.get_resources_info()
have_gpu = False if local_resources[7] is None else True
if have_gpu: gpu_name = local_resources[7][0][1]
# print(gpu_name)
if have_gpu:
    #gpu_info.append([gpu_id, gpu_name, gpu_memory_total, gpu_memory_free, gpu_load])
    if len(local_resources[7]) > 1:
        pass #TODO: address multiple GPU cases
    else:
        gpu_name, gpu_memory_total, gpu_memory_free = local_resources[7][0][1], local_resources[7][0][2], local_resources[7][0][3]
else:
    mem_total, mem_available = local_resources[5], local_resources[6]
print(f'\nMeasuring Local Resources Finish!\n{"="*110}\n\n')

# scrape.py
print(f'{"="*110}\nScraping HuggingFace open-llm-leaderboard (https://huggingfaceh4-open-llm-leaderboard.hf.space)...')
scrape.scrape() # update leaderboard
print(f'\nScraping HuggingFace open-llm-leaderboard Finish!\nSee best-models.txt and best-models-deduplicate.txt for best models for each size and each kind.\n{"="*110}\n\n')

'''
Optional Method: using second-hand API
'''
# from gradio_client import Client
# import json
# client = Client("https://felixz-open-llm-leaderboard.hf.space/")

# json_data = client.predict("","", api_name='/predict')

# with open(json_data, 'r') as file:
#     file_data = file.read()

# # Load the JSON data
# data = json.loads(file_data)

# # Get the headers and the data
# headers = data['headers']
# data = data['data']

print(f'{"="*110}\nData Cleaning starts...')
# load leaderboard data
data = pd.read_csv('open-llm-leaderboard.csv') # read the full leaderboard

# data cleaning

# fixing a mistake for microsoft's phi-1_5
data.loc[data['model_name_for_query']=='microsoft/phi-1_5', '#Params (B)'] = 1.3
# fixing a mistake for roneneldan/TinyStories-1M
data.loc[data['model_name_for_query']=='roneneldan/TinyStories-1M', '#Params (B)'] = 0.001
# removing models with size 0
data = data.loc[data['#Params (B)'] != 0]
# change the type of lamini 774M to Finetuned
data.loc[data['model_name_for_query']=='MBZUAI/LaMini-GPT-774M', 'Type'] = "fine-tuned"
print(f'\nData Cleaning Finish!\n{"="*110}\n\n')
# remove not available models
data = data.loc[data['Available on the hub'] == True]

# Prompt the user to input something
# TODO: Refine the logic: skip character and more while loops
print(f'{"="*110}\nPlease input your desired task types...')
model_type = input("Model type: ")
while (model_type not in {'pretrained', 'fine-tuned', 'instruction-tuned', 'RL-tuned', 'all'}):
    model_type = input("Invalid input. Please pick from {'pretrained', 'fine-tuned', 'instruction-tuned', 'RL-tuned', 'all'}.\nModel type: ")
num_likes = input("Minimum number of likes: ")
try:
    num_likes = int(num_likes)
except Exception as e:
    print("Your input is not integet. Will proceed with minimum number of likes being 0.")
    num_likes = 0
desired_mem_usage = input("How many percantages of your available memory do you want to use? (0,100) :")
try:
    desired_mem_usage = float(desired_mem_usage)
except Exception as e:
    print("Your input is not a number. Will proceed with recommended desired_mem_usage being 50%.")
    desired_mem_usage = 50
if desired_mem_usage <= 0 or desired_mem_usage > 100:
    print("Your input is not a valid percentage. Will proceed with recommended desired_mem_usage being 50%.")
    desired_mem_usage = 50
metrics = input('What benchmark dataset do you want your model to excel at?\nSupported metrics include Average, ARC, HellaSwag, MMLU, TruthfulQA, Winogrande, GSM8K : ')
if not metrics in {'Average', 'ARC', 'HellaSwag', 'MMLU', 'TruthfulQA', 'Winogrande', 'GSM8K'}:
    print("Your input metrics is not supported. Will proceed with recommended metrics Average.")
    metrics = 'Average ⬆️'
if metrics == "Average": metrics = 'Average ⬆️'
train_or_inference = input("Train or Inference: ")
while (train_or_inference not in {'inf', 'inf_vLLM', 'inf_ggml', 'trn'}):
    train_or_inference = input("Invalid input. Please pick from {'inf', 'inf_vLLM', 'inf_ggml', 'trn'}.\nTrain or Inference: ")
if train_or_inference == 'trn':
    train_method = input("Train method: ")
    while (train_method not in {'full_trn','lora_trn','qlora'}):
        train_method = input("Invalid input. Please pick from {'full_trn','lora_trn','qlora'}.\nTrain method: ")
    optimizer = input("Optimizer: ")
    while (optimizer not in {'adam_opt', 'sgd_opt'}):
        optimizer = input("Invalid input. Please pick from {'adam_opt', 'sgd_opt'}.\nOptimizer: ")
    gradient_checkpointing = True if input("Gradient checkpointing? {'y', 'n'} ")=="y" else False
    quant, prompt_len, tokens_to_generate = None, None, 1
else: # inference
    # speed = input("What is the lowest tokens/s you can accept: ") # tokens per second
    quant = input("Quantization method: ")
    while (quant not in {'no_quant', 'bnb_int8', 'bnb_q4', 'ggml_Q2_K', 'ggml_Q3_K_L','ggml_Q3_K_M', 'ggml_QK4_0','ggml_QK4_1','ggml_QK4_K_M','ggml_QK4_K_S', 'ggml_QK5_0', 'ggml_QK5_1', 'ggml_QK5_K_M', 'ggml_Q6_K', 'ggml_QK8_0'}):
        quant = input("Invalid input. Please pick from {'no_quant', 'bnb_int8', 'bnb_q4', 'ggml_Q2_K', 'ggml_Q3_K_L','ggml_Q3_K_M', 'ggml_QK4_0','ggml_QK4_1','ggml_QK4_K_M','ggml_QK4_K_S', 'ggml_QK5_0', 'ggml_QK5_1', 'ggml_QK5_K_M', 'ggml_Q6_K', 'ggml_QK8_0'}.\nQuantization method: ")
    try:
        prompt_len = int(input("Prompt length in tokens: "))
        tokens_to_generate = int(input("Output length in tokens: "))
    except Exception as e:
        print(f'prompt_len and output_len should be positive int. We will proceed with default value:{300, 300}')
        prompt_len, tokens_to_generate = 300, 300
    train_method, optimizer, gradient_checkpointing = None, None, None
batch_size = int(input("Batch Size: ")) # modify
print(f'\nBased on your input, the task variables are \n\
      model_type: {model_type}, \n\
      minimum number of likes: {num_likes}, \n\
      desired memory usage: {desired_mem_usage}, \n\
      train_or_inference: {train_or_inference}, \n\
      train_method: {train_method},\n\
      optimizer: {optimizer}, \n\
      gradient_checkpointing: {gradient_checkpointing},\n\
      quant: {quant}, \n\
      prompt_len: {prompt_len}, \n\
      tokens_to_generate: {tokens_to_generate}, \n\
      batch_size: {batch_size} \n\
      \n{"="*110}\n\n')

print(f'{"="*110}\nData Augmentation using get_model_memory starts...')
# Wrapper function for applying get_model_memory with error handling
def apply_get_model_memory(row):
    try:
        return get_model_memory.findMemoryRequirement(int(row['#Params (B)']), train_or_inference, train_method, optimizer, quant, prompt_len, tokens_to_generate, batch_size, gradient_checkpointing)['Total']
    except Exception as e:
        # assign a negative value as indicator
        return -1

# Apply the function to each row and create a new column
data['Memory'] = data.apply(apply_get_model_memory, axis=1)

data = data.loc[data['Memory'] != -1] # TODO: give estimates to models not in the list
if model_type != 'all': data = data.loc[data['Type'] == model_type]
if num_likes > 0: data = data.loc[data['Hub ❤️'] >= num_likes]

print(f'\nData Augmentation Finish!\n{"="*110}\n\n')

def find_best_model(models):
    best_model_names = set()
    for i in range(len(seperation)+1):
        low = seperation[i-1] if i != 0 else 0
        high = seperation[i] if i != len(seperation) else None
        sub_models = models[models['#Params (B)'] >= low]
        if high: sub_models = sub_models[sub_models['#Params (B)'] < high] # if high is not None
        if len(sub_models) == 0: continue # skip if there are no models within this size range
        max_score_index = sub_models['Average ⬆️'].idxmax()
        print(f'({low},{high})', sub_models.loc[max_score_index]["model_name_for_query"])
        best_model_names.add(((low, high), sub_models.loc[max_score_index]["model_name_for_query"]))
    return best_model_names

print(f'{"="*110}\nSearching for best models...')
if have_gpu:
    deployable_models = data[data['Memory'] < gpu_memory_free*1024*desired_mem_usage*0.01]
else:
    deployable_models = data[data['Memory'] < mem_available*1024*desired_mem_usage*0.01]

try:
    max_score_index = deployable_models[metrics].idxmax()
except ValueError as e:
    print("There are no qualifying models based on your preferences. Loosening the requirements and try again!")
    exit()
model_name = deployable_models.loc[max_score_index]["model_name_for_query"]
repo_address = deployable_models.loc[max_score_index]["Model_repo"]
print(f'\nBest Model Found! Model name: {model_name}. Model repo: {repo_address}.\n{"="*110}\n\n')

auto = input("Do you want to try downloading and deploying this model automatically? : ")
while (auto.lower() not in {'y', 'yes', 'n', 'no'}):
    auto = input("Invalid input. Please pick from {'Y', 'N'}.\nDo you want to try downloading and deploying this model automatically? : ")
if auto.lower() == "y" or auto.lower() == "yes":
    # model_name = model_name.replace('/', '@')
    # print(model_name)
    # model_name
    # data.loc[data['Model_name_for_query'] == "microsoft/phi-1_5"]

    # exit()
    # os.chdir(f'Automation/{model_name}')
    os.chdir(f'Automation/general')
    subprocess.run(['pip', 'install', '-r', 'requirements.txt'])
    subprocess.run(['python', 'init.py', model_name])
else:
    print(f"Thank you for using choose-your-llm.\nOur recommended model based on your info is: {model_name}\n You can go this {repo_address} to see more details. :)")

