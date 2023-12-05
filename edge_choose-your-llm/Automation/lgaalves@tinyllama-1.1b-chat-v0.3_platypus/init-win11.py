#!/usr/bin/env python
# coding: utf-8

# In[2]:


from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import tensorflow as tf
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

model_name_or_path = "TheBloke/tinyllama-1.1b-chat-v0.3_platypus-GPTQ"

model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

prompt = "Tell me about AI"
prompt_template=f'''Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:
'''

print("\n\n*** Generate:")

input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
print(tokenizer.decode(output[0]))

