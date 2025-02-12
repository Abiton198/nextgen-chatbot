from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "facebook/blenderbot-400M-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

input_text = "Hello, how are you?"
inputs = tokenizer(input_text, return_tensors="pt")

output = model.generate(**inputs)
print(tokenizer.decode(output[0], skip_special_tokens=True))
