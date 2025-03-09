import logging
import datasets
import torch
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def load_dataset_sst2(batch_size=8, train_subset=5000, val_subset=None):
    """
    Loads the SST-2 dataset from huggingface 'glue' and returns a PyTorch DataLoader.
    Subset the training data to 'train_subset' examples for faster training.

    Args:
        batch_size (int): batch size for DataLoaders
        train_subset (int): how many training examples to keep. Use None for all.
        val_subset (int): how many validation examples to keep. Use None for all.
    """
    logging.info("Loading SST-2 dataset...")
    dataset = datasets.load_dataset("glue", "sst2")
    
    train_data = dataset["train"]        # ~67k
    val_data   = dataset["validation"]   # ~0.87k

    # Subset if desired
    if train_subset is not None and train_subset < len(train_data):
        logging.info(f"Reducing training set to {train_subset} samples.")
        train_data = train_data.select(range(train_subset))
    if val_subset is not None and val_subset < len(val_data):
        logging.info(f"Reducing validation set to {val_subset} samples.")
        val_data   = val_data.select(range(val_subset))

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    def tokenize_fn(examples):
        return tokenizer(examples["sentence"], 
                         padding="max_length", 
                         truncation=True, 
                         max_length=128)
    
    train_data = train_data.map(tokenize_fn, batched=True)
    val_data   = val_data.map(tokenize_fn,   batched=True)
    
    train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_data.set_format(type="torch",   columns=["input_ids", "attention_mask", "label"])
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_data,   batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, tokenizer
