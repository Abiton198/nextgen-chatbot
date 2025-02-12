from transformers import ConversationalPipeline, Conversation, AutoModelForCausalLM, AutoTokenizer

model_name = "facebook/blenderbot-400M-distill"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

chat = ConversationalPipeline(model=model, tokenizer=tokenizer)

conversation = Conversation("Hello, how's your day?")
chat([conversation])

print(conversation)
