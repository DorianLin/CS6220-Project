import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from conversation import get_default_conv_template

# MiniChat
tokenizer = AutoTokenizer.from_pretrained("GeneZC/MiniChat-1.5-3B", use_fast=False)
# GPU.
# model = AutoModelForCausalLM.from_pretrained("GeneZC/MiniChat-1.5-3B", use_cache=True, device_map="auto", torch_dtype=torch.float16).eval()
# CPU.
model = AutoModelForCausalLM.from_pretrained("GeneZC/MiniChat-1.5-3B", use_cache=True, device_map="cpu", torch_dtype=torch.float32).eval()

conv = get_default_conv_template("minichat")

question = "Implement a program to find the common elements in two arrays without using any extra data structures."
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
input_ids = tokenizer([prompt]).input_ids
output_ids = model.generate(
    torch.as_tensor(input_ids),
    do_sample=True,
    temperature=0.7,
    max_new_tokens=300,
)
output_ids = output_ids[0][len(input_ids[0]):]
output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
print(question, "\n")
print(prompt, "\n")
print(output)

while True:
    question = input("Please input your prompt: ")
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer([prompt]).input_ids
    output_ids = model.generate(
        torch.as_tensor(input_ids),
        do_sample=True,
        temperature=0.7,
        max_new_tokens=100,
    )
    output_ids = output_ids[0][len(input_ids[0]):]
    output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    print(output)
# output: "def common_elements(arr1, arr2):\n    if len(arr1) == 0:\n        return []\n    if len(arr2) == 0:\n        return arr1\n\n    common_elements = []\n    for element in arr1:\n        if element in arr2:\n            common_elements.append(element)\n\n    return common_elements"
# Multiturn conversation could be realized by continuously appending questions to `conv`.

