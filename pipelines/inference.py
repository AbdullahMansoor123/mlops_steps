import torch


def inference(model, tokenizer, text, device):
    """
    Perform inference using the trained model on a new text input.
    """
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs).logits
    prediction = torch.argmax(outputs, dim=1).item()
    return prediction


# Perform inference on sample text
test_text = "This movie was fantastic! I really enjoyed it."

print(f"Inference Result: {inference(model, tokenizer, test_text, device)}")