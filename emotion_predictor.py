import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model and tokenizer
model_name = "joeddav/distilbert-base-uncased-go-emotions-student"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Emotion labels
id2label = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

# Input text
text = "I passed the exam and I feel so relieved and happy!"


# Tokenize
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

# Predict
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits)  # multi-label probs
    top_idx = torch.argmax(probs, dim=1).item()  # single top emotion

# Show result
print(f"Predicted emotion: {id2label[top_idx]}")
