import torch


def train_model(train_loader, val_loader, model, optimizer, device, epochs=3):
    """
    Train the model using the provided data loaders, optimizer, and device.
    """
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total_loss, total_correct = 0, 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels).logits

            loss = loss_fn(outputs, labels)
            loss.backward()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")
    

    # Validation loop
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels).logits
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    print(f"Validation Accuracy: {correct / total * 100:.2f}%")


