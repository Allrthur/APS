# builtin
import datetime
import os
# project
from load_dataset import load_dataset
import utils
# installed
import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, DataCollatorForLanguageModeling
from transformers import pipeline
from datasets import Dataset
from accelerate import Accelerator
from tqdm import tqdm, trange

# Function to fine-tune model before feature extraction
def domain_adapt_model(
    train: pd.DataFrame,
    model_name: str,
    batch_size: int = 12,
    random_state: int = 1,
    epochs:int =3,
    ft_model_name: str = ""
) -> AutoModelForMaskedLM:
    """
    Fine-tune a Masked Language Model using Hugging Face transformers.
    
    Args:
        train: DataFrame with "text" column containing training sentences
        model_name: Pretrained model name (e.g., "bert-base-uncased")
        batch_size: Batch size for training
        random_state: Seed for reproducibility
    
    Returns:
        Fine-tuned model for MLM task
    """
    # Set seeds for reproducibility
    utils.lock_seed(random_state)

    # Validate input format
    assert "text" in train.columns

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    # Convert DataFrame to Hugging Face Dataset
    dataset = Dataset.from_pandas(train[["text"]], preserve_index=False)

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=120,
            return_special_tokens_mask=True
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    # Create DataLoader
    train_loader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator
    )

    # Initialize Accelerator
    accelerator = Accelerator()
    model, optimizer, train_loader = accelerator.prepare(
        model,
        torch.optim.AdamW(model.parameters(), lr=1e-5),
        train_loader
    )

    # # Rid memory off non important bits
    # del tokenized_dataset
    # del dataset

    # # Training setup
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # Training loop
    model.train()
    t = trange(epochs, desc=f"Avg Loss= ... ", leave=True)
    for epoch in t:
        total_loss = 0
        for batch in train_loader:
            outputs = model(**batch)
            loss = outputs.loss
            
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        t.set_description(f"Avg Loss= {avg_loss:1.2f}")

    # Save model weights
    if ft_model_name == "":
        ft_model_name = f"models/{model_name.split('/')[-1]}_{datetime.datetime.now().strftime('%d-%m-%YT%H%M')}"

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(ft_model_name)
    tokenizer.save_pretrained(ft_model_name)

    # Delete model from memory
    del model
    
    return ft_model_name

def extract_cls_token(
    df: pd.DataFrame,
    batch_size: int = 16,
    max_length: int = 128,
    fold_name: str | None = None,
    fold_id: int | None = None,
    random_state: int = 1,
    model_name: str = "FacebookAI/xlm-roberta-base",
    dataset_name: str = "default_dataset",  # Added to prevent undefined variable error
    bypass_cache: bool = False
) -> list:
    
    # 1. Check the cache
    path = None
    if fold_name is not None and fold_id is not None and not bypass_cache:
        print("Procurando fts salvas...")
        # Clean model name to create a valid directory path (e.g., "FacebookAI_xlm-roberta-base")
        safe_model_name = model_name.replace("/", "_")
        directory = f"extracted_fts/cls/{safe_model_name}/{dataset_name}"
        path = f"{directory}/{fold_name}_fold{fold_id}.pt"
        
        if os.path.exists(path):
            print("Cache hit! Carregando features...")
            return torch.load(path, weights_only=True)
        else:
            print("Cache miss.")

    # 2. Instantiate the pipeline
    # Note: Added device=0 if GPU is available to speed up inference, and passed the batch_size
    device = 0 if torch.cuda.is_available() else -1
    extractor = pipeline(
        model=model_name,
        task="feature-extraction",
        padding="max_length",
        max_length=max_length,
        device=device,
        batch_size=batch_size
    )
    
    # 3. Extract features
    print("Extraindo tokens CLS...")
    # Passing a dataset/list to pipeline returns a generator or list of tensors
    featurelist = extractor(df["text"].tolist())
    
    # 4. Process and isolate the CLS token
    # Hugging Face feature-extraction returns a list of lists of floats or tensors.
    # The CLS token is always the first token: shape [1, sequence_length, hidden_dim] -> index [0][0]
    cls_tokens = []
    for out in featurelist:
        # Convert to tensor if it isn't already, squeeze batch dim if present
        tensor_out = torch.tensor(out)
        # Assuming shape is [1, seq_len, hidden_dim] or [seq_len, hidden_dim]
        if tensor_out.ndim == 3:
            cls_token = tensor_out[0, 0, :]
        else:
            cls_token = tensor_out[0, :]
        cls_tokens.append(cls_token)
        
    # 5. Save to cache if path was defined
    if path:
        print(f"Salvando features em: {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(cls_tokens, path)
        
    return cls_tokens

def extract_mean_pooled_embeddings(
    df: pd.DataFrame,
    batch_size: int = 16,
    max_length: int = 128,
    fold_name: str | None = None,
    fold_id: int | None = None,
    model_name: str = "FacebookAI/xlm-roberta-base",
    dataset_name: str = "default_dataset",
    bypass_cache: bool = False
) -> torch.Tensor:
    
    # 1. Check Cache
    path = None
    if fold_name is not None and fold_id is not None and not bypass_cache:
        print("Procurando fts salvas...")
        safe_model_name = model_name.replace("/", "_")
        directory = f"extracted_fts/mean_pooled/{safe_model_name}/{dataset_name}"
        path = f"{directory}/{fold_name}_fold{fold_id}.pt"
        
        if os.path.exists(path):
            print("Cache hit! Carregando features...")
            return torch.load(path, weights_only=True)
        else:
            print("Cache miss.")

    # 2. Setup Device, Tokenizer, and Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()  # Put model in evaluation mode

    texts = df["text"].tolist()
    all_embeddings = []

    print("Extraindo embeddings com Mean Pooling...")
    
    # 3. Batch Processing
    with torch.no_grad():  # Disable gradient calculation for speed/memory
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            
            # Tokenize the batch
            encoded_input = tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)
            
            # Compute token embeddings
            model_output = model(**encoded_input)
            
            # Perform Mean Pooling safely ignoring padding
            token_embeddings = model_output.last_hidden_state  # Shape: [batch, seq_len, hidden_dim]
            attention_mask = encoded_input["attention_mask"]   # Shape: [batch, seq_len]
            
            # Expand attention mask to match token embeddings dimensions
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            
            # Sum embeddings along sequence length dimension, ignoring padding
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            
            # Clamp the sum of the mask to avoid dividing by zero on empty sequences
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            
            # Divide sum by real token counts
            batch_mean_embeddings = sum_embeddings / sum_mask
            
            # Move to CPU before storing to preserve GPU memory
            all_embeddings.append(batch_mean_embeddings.cpu())

    # Concatenate all batch tensors into a single tensor
    final_embeddings = torch.cat(all_embeddings, dim=0)

    # 4. Save to Cache
    if path:
        print(f"Salvando features em: {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(final_embeddings, path)
        
    return final_embeddings

# TODO: Create feature extraction pre-processing step with caching
if __name__ == "__main__":
    
    MODEL = "FacebookAI/xlm-roberta-base"
    FTS_OUTPUT_DIR = f"extracted_fts/{MODEL}/"
    datasets = [
        "unbalanced/sst2",
        "unbalanced/semeval2017",
        "unbalanced/sst2",
        "unbalanced/sst2",
        "semeval2015",
        "semeval2024",
        "semeval2025_eng_anger",
        "twitter_topics_0_a",
        "twitter_topics_1_a",
    ]
    
    # Caching fts
    for dataset in datasets:
        # Setting up
        for fold_id in range(10):
            train_fold = pd.read_csv(f"dataset/saved_folds/{dataset}/train_fold{fold_id}.csv")
            val_fold = pd.read_csv(f"dataset/saved_folds/{dataset}/val_fold{fold_id}.csv")
            test_fold = pd.read_csv(f"dataset/saved_folds/{dataset}/test_fold{fold_id}.csv")
            
            # Extrct CLS token
            cls = extract_mean_pooled_embeddings(
                df=train_fold,
                batch_size=16,
                max_length=128,
                fold_name="train",
                fold_id=fold_id,
                dataset_name=dataset
            )
            print(cls)