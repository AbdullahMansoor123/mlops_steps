This a guide for mlops engineer who want to learn mlops from basic to advance in easy way.
- Data.py the data pipelines work like this:
    1. load the data from the hugging face 
    2. split the data into train, val and test set
    3. Since we are working on the text data we need to tokenize the data using the tokenizer from hugging face
    4. format the data in torch dataset format as we are using pytorch for training the model
    5. create the dataloaders for train, val and test set
- Model.py
    1. optional(create the model from scratch)
    2. load the model from hugging face 
- Train.py
    1. 
- inference.py 