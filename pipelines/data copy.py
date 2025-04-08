from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer



class DataPrepModule:
    def __init__(self,dataset_name,model_name ,max_length, batch_size=8):
        # self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def prepare_data(self):
        dataset = load_dataset(self.dataset_name)
        # Split into train, validation, and test sets
        self.train_data = dataset['train']
        self.val_data = dataset['test']
        self.test_data = dataset['unsupervised']

    def tokenize_data(self, data):
        return self.tokenizer(data['text'],
                                padding="max_length", 
                                truncation=True, 
                                max_length=self.max_length)

    def format_data(self, stage=None):
        if stage == "fit" or stage is None:  
            self.train_encodings = self.train_data.map(self.tokenizer, batched=True)
            self.train_encodings.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

            self.val_encodings = self.val_data.map(self.tokenizer, batched=True)
            self.val_encodings.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

            self.test_encodings = self.test_data.map(self.tokenizer, batched=True)
            self.test_encodings.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    
    def train_dataloaders(self):
        return DataLoader(
            self.train_encodings, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloaders(self):
        return DataLoader(
            self.val_encodings, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloaders(self):
        return DataLoader(
            self.test_encodings, batch_size=self.batch_size)





if __name__ == "__main__":
    data_model = DataPrepModule(dataset_name='imdb', model_name="bert-base-uncased", max_length=512, batch_size=8)
    data_model.prepare_data()
    data_model.format_data()
    batch = next(iter(data_model.train_dataloaders()))
    print(batch['sentence'])
