import torch
from transformers import BertTokenizerFast, BertForSequenceClassification

# Use a publicly available GoEmotions model
model_name = "monologg/bert-base-cased-goemotions-original"

# Load tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Put model on device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Emotion label mapping (27 emotions + neutral)
id2label = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

# Enter your custom text here
test_text = "i am happy."

# Tokenize input
inputs = tokenizer(test_text, return_tensors="pt", truncation=True, padding=True).to(device)

# Run prediction
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=1).item()

# Print result
print(f"Predicted emotion: {id2label[predicted_label]}")
