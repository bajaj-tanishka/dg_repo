
import torch
from dgs_model import load_dgs_model

# Load all three return values
model, labels, config = load_dgs_model()

print("Checkpoint loaded successfully.")
print("Model:", model.__class__.__name__)
print("Number of classes:", len(labels))
print("Labels:", labels)

# Read expected dimensions from the configuration
architecture = config["architecture"]
sequence_length = config.get("sequence_length", 30)
input_dim = architecture["input_dim"]

print("\nExpected sequence length:", sequence_length)
print("Expected features per frame:", input_dim)

# Create test input: (batch, frames, features)
sample = torch.randn(1, sequence_length, input_dim)

model.eval()

with torch.no_grad():
    output = model(sample)

print("\nInference test successful!")
print("Input shape:", tuple(sample.shape))
print("Output shape:", tuple(output.shape))

assert output.shape == (1, len(labels)), (
    f"Unexpected output shape: {tuple(output.shape)}"
)

probabilities = torch.softmax(output, dim=1)
predicted_index = probabilities.argmax(dim=1).item()

print("Predicted label:", labels[predicted_index])
print(f"Confidence: {probabilities[0, predicted_index].item():.4f}")
print("\nAll model shape checks passed.")
