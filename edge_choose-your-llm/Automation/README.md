# Organization of Automation Folder

## Intro

In this folder, we will create separate folders for each model and inside the folder we will put the following files:

- Model Description markdown: put huggingface link and exact model size here, one per model
- Script to download and set up the environment as well as a test run, one per environment(mac, windows-gpu-8G, etc)
- requirements-{machine}.txt, some common requirements are transformers and torch
- Example: MBZUAI@LaMini-Neo-125M