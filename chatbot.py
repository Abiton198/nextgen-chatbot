from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load DistilBERT model and tokenizer
model_name = "distilbert-base-uncased"
model = DistilBertForSequenceClassification.from_pretrained(model_name)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)

input_text = "Hello, how are you?"

# Tokenize input
inputs = tokenizer(input_text, return_tensors="pt")

# Forward pass through the model
outputs = model(**inputs)

# Process the output (e.g., get the logits)
logits = outputs.logits
print(logits)


# TO DO
# - install python 3.10 version
# - re deploy in Render
