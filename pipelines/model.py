import torch
from transformers import AutoModelForSequenceClassification

def load_model(model_name, num_classes):
    """
    Load a pre-trained model for sequence classification.
    """    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)

    return model


# # Test model function
# model_name= "bert-base-uncased"
# model = load_model(model_name= model_name,
#                    num_classes=2)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(model.to(device))

