from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
import wandb
from wandb.wandb_run import Run
from typing import Union
from huggingface_hub import login

class SequenceClassifierWrapper:
    def __init__(self, 
        model_name: str, 
        num_labels: int = 2,
        random_state: int = 1,
        batch_size: int = 32,
        epochs: int = 10,
        lr: int = 2e-5,
        # Early stop settings
        minimum_delta:float = 0.001,
        patience:int = 3,
        freeze_lm: int = False,
        # Tokenizer args
        max_length: int = 120,
        # Wandb run
        wandb_run:Run = None,
        # Push to Hub
        trained_model_name:str = "testmodel",
        save_model:bool = False,
    ):
        self.trained_model_name = trained_model_name
        # Data for loading model
        self.model_name = model_name
        self.random_state = random_state
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.seed = random_state
        self.max_length = max_length
        # Early stop 
        self.minimum_delta = minimum_delta
        self.patience = patience
        # Data for wandb
        self.wandb_run = wandb_run
        # Freeze lm
        if freeze_lm: 
            if model_name=="jhu-clsp/bernice":
                for param in self.model.roberta.parameters(): 
                    param.requires_grad = False
            elif model_name=="Twitter/twhin-bert-base":
                for param in self.model.bert.parameters():
                    param.requires_grad = False
            else:
                raise Exception(f"{model_name} is an unsuported model for freezing lm")
        # Variables for training
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.save_model = save_model

    def __preprocess_dataset(self, X:pd.Series, y:List[int]=None)->Dataset:
        if not y: y = [0]*len(X)
        dataset:Dataset = Dataset.from_pandas(pd.DataFrame({"text":X.to_list(),"labels":y}))
        # Tokenize
        def tokenize_function(example: Dict[str, List]):
            return self.tokenizer(example, padding="max_length", truncation=True, max_length=self.max_length)
        dataset = dataset.map(tokenize_function, input_columns=["text"], batched=True)
        # Convert to PyTorch format
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return dataset

    def predict(self, X: Union[str, list[str], Dataset, pd.Series, pd.DataFrame])->list[int]:
        # If input is a DataFrame, convert it into a Dataset
        if type(X)==pd.Series:
            ds = self.__preprocess_dataset(X)
        else:
            print("You are currently using an unsupported type please convert it to pd.Series")
        
        loader: DataLoader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        # Feed model inputs
        res = []
        for batch in loader:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            # Forward pass
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(**batch)
            # Get the predicted class
            logits = outputs.logits
            predicted_classes:torch.Tensor = torch.argmax(logits, dim=1).tolist()
            res.extend(predicted_classes)
        # Prepare the result
        return res

    def fit(self, 
        X: pd.Series, 
        y: List[int],
        val_X: pd.Series = None,
        val_y: List[int] = None
    ):
        # Convert into Dataset
        train_ds:Dataset = self.__preprocess_dataset(X, y)
        val_ds: Dataset = self.__preprocess_dataset(val_X, val_y)
        # Set early stop params
        early_stop_args: EarlyStoppingCallback = EarlyStoppingCallback(
            early_stopping_patience=self.patience,
            early_stopping_threshold=self.minimum_delta
        )
        # Set training params
        training_args : TrainingArguments = TrainingArguments(
            # Main Params
            output_dir=f"models/{self.model_name}",
            num_train_epochs=self.epochs,
            # Data
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            # Optimizer params
            learning_rate=self.lr,
            weight_decay=0.1,
            warmup_steps=500,
            # Eval params
            eval_strategy="epoch",
            # eval_steps=len(train_ds),
            metric_for_best_model="loss",
            greater_is_better=False,
            # Saving strategy params
            save_strategy="epoch",
            # save_steps=len(train_ds),
            load_best_model_at_end=True,
            save_total_limit=5,
            # Misc
            seed=self.random_state,
            # Wandb
            run_name=None,
            logging_strategy="epoch",
            logging_first_step=True

        )
        # Use huggingface
        trainer: Trainer = Trainer(
            model=self.model, 
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            callbacks=[early_stop_args],
        )
        trainer.train()
        # Update model
        self.model = trainer.model
        # try:
        if self.save_model:
            try:
                self.model.push_to_hub(repo_id=self.trained_model_name, private=False)
                self.tokenizer.push_to_hub(repo_id=self.trained_model_name, private=False)
            except Exception as e:
                print("Could not push model to hub:", e)
                self.model.save_pretrained(f"models/manually_saved/{self.trained_model_name}")
                self.tokenizer.save_pretrained(f"models/manually_saved/{self.trained_model_name}")
        # except:
            # print("model failed to save")


# Example usage
if __name__ == "__main__":
    from datasets import Dataset
    # from prep_dataset import split_train_val_test
    # from load_dataset import load_dataset

    classifier = SequenceClassifierWrapper(
        "bert-base-uncased", 
        num_labels=2, 
        epochs=1,
        save_model=True,
        trained_model_name="test_model"
    )
    classifier.fit(
        X=pd.Series([
            "Uma noite em 1001",
            "AAAAAAAA",
            "SUCESSO"
        ]),
        y=[1,0,0],
        val_X=pd.Series(["UUUUU", "Setecentos e oitenta"]),
        val_y=[0,1]
    )