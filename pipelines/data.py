from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer



class DataModule:
    def __init__(self, dataset_name='imdb', model_name="bert-base-uncased", max_length=512, batch_size=8, reduce_fraction = 0.10):
        # self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.reduce_fraction = reduce_fraction
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.prepare_data(dataset_name)
        self.prepare_dataloader()
    
    def prepare_data(self, dataset_name):
        dataset = load_dataset(dataset_name)

        def reduce_split(split):
            reduced_size = int(len(split) * self.reduce_fraction)
            return split.select(range(reduced_size))

        self.train_data = reduce_split(dataset['train'])
        self.val_data = reduce_split(dataset['test'].select(range(len(dataset['test']) // 2)))
        self.test_data = reduce_split(dataset['unsupervised'].select(range(len(dataset['test']) // 2, len(dataset['test']))))

        self.train_encodings = self.train_data.map(self.tokenize_data, batched=True)
        self.train_encodings.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

        self.val_encodings = self.val_data.map(self.tokenize_data, batched=True)
        self.val_encodings.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

        self.test_encodings = self.test_data.map(self.tokenize_data, batched=True)
        self.test_encodings.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    def tokenize_data(self, data):
        return self.tokenizer(data['text'],
                                padding="max_length", 
                                truncation=True, 
                                max_length=self.max_length)

    
    def prepare_dataloader(self):
        self.train_loader = DataLoader(
            self.train_encodings, batch_size=self.batch_size, shuffle=True)
    
        self.val_loader = DataLoader(
            self.val_encodings, batch_size=self.batch_size, shuffle=False)
    
        self.test_loader = DataLoader(
            self.test_encodings, batch_size=self.batch_size)


# for testing the DataModule class

# if __name__ == "__main__":
#     data_model = DataModule()
#     data_model.prepare_data(dataset_name='imdb')
#     batch = next(iter(data_model.train_loader))
#     print(batch)
