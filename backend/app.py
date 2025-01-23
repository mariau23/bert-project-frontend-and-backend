from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer
from transformers import BertForSequenceClassification
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow CORS for your React frontend's origin
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the tokenizer (this can still use Hugging Face's pre-trained tokenizers)
model_name = "bert-base-uncased"  # Use the tokenizer associated with your model architecture
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define your model architecture
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)

# Load the saved model state dictionary
model_checkpoint_path = r"C:\test\model_checkpoint3_epoch7.pth"  # Path to your saved model checkpoint
state_dict = torch.load(model_checkpoint_path)

# Remove classifier weights and biases if they exist
state_dict.pop("classifier.weight", None)
state_dict.pop("classifier.bias", None)

# Load the remaining state_dict into the model
model.load_state_dict(state_dict, strict=False)

# Move model to GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Define a request schema for inputs
class TextInput(BaseModel):
    text: str

# Define labels based on your model's output
label_mapping = {0: "Positive", 1: "Negative", 2: "Neutral", 3: "Irrelevant"}

# Route for the prediction
@app.post("/predict")
async def predict_sentiment(input: TextInput):
    # Tokenize the input text
    inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get the model's prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        # Apply softmax to convert logits into probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class = torch.argmax(logits, dim=-1).item()

    # Map the prediction to a human-readable label
    sentiment = label_mapping[predicted_class]

    # Get the probability score for the predicted class
    score = probs[0][predicted_class].item()

    return {"text": input.text, "sentiment": sentiment}
