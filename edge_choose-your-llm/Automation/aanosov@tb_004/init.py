# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="aanosov/tb_004")

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("aanosov/tb_004")
model = AutoModelForCausalLM.from_pretrained("aanosov/tb_004")
