from transformers import pipeline
import sys
# Use a pipeline as a high-level helper

try:

    pipe = pipeline("text-generation", model=sys.argv[1])
    input_prompt = f"The president of USA is: "
    generated_text = pipe(input_prompt, max_length=10, do_sample=True)[0]['generated_text']
    print("Response", generated_text)

    while True:
        inputs = input("Please input your prompt: ")
        generated_text = pipe(inputs, max_length=100, do_sample=True)[0]['generated_text']
        print("Response:\n",generated_text)
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    print(f"Try loading the model directly")
    try: 
        # # Load model directly
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])
        model = AutoModelForCausalLM.from_pretrained(sys.argv[1])
        inputs = tokenizer("The president of USA is: ", return_tensors="pt", return_attention_mask=False)
        outputs = model.generate(**inputs, max_length=10)
        text = tokenizer.batch_decode(outputs)[0]
        print(text)

        while True:
            prompt = input("Please input your prompt: ")
            inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
            outputs = model.generate(**inputs, max_length=100)
            text = tokenizer.batch_decode(outputs)[0]
            print(text)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print(f"Try loading the model through adapters.")
        pass
finally:
    print("Please look into the model card for more info!")

