# Builtin
from typing import List, Literal, Union
import re
import os
import time
import json
import argparse
import gc
import statistics
# Local Imports
from load_dataset import load_dataset
# Installed Libs
import torch
import pandas as pd
import numpy as np
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel
from collections import Counter
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA, DictionaryLearning
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import Counter
from tqdm import tqdm

class Selector:
    def __init__(self, dataset: pd.DataFrame, paraphrases: pd.DataFrame, model_name: str = "FacebookAI/xlm-roberta-base"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        # Validation
        if not {"text", "label"}.issubset(dataset.columns):
            raise ValueError("dataset must have 'text' and 'label' columns")
        if not {"original", "paraphrase"}.issubset(paraphrases.columns):
            raise ValueError("paraphrases must have 'original' and 'paraphrase' columns")
            
        self.dataset = dataset
        self.paraphrases = paraphrases[paraphrases["original"].isin(dataset["text"])].copy()
        
        # Pre-compute original embeddings to find prototypes
        self.original_embeddings = self._embed(dataset["text"].tolist())
        self.dataset_embeddings = dict(zip(dataset["text"], self.original_embeddings))

    def _embed(self, texts, batch_size=16):
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Batch Embedding"):
            batch_texts = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                return_tensors="pt", 
                max_length=128
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Mean Pooling: mean of all token embeddings, excluding padding
                mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                batch_embs = sum_embeddings / sum_mask
                
            all_embeddings.append(batch_embs.cpu().numpy())
            
        return np.vstack(all_embeddings)

    def select(self, n_samples: int, target_label:int):
        raise NotImplementedError("Subclasses must implement select()")

class FirstKSelector(Selector):
    def __init__(self, 
                 dataset: pd.DataFrame, 
                 paraphrases: pd.DataFrame,
                 model_name:str,
                ):
        """
        Initializes the FirstKSelector without loading embedding models.
        """
        # 1. Retain the necessary Validation
        if not {"text", "label"}.issubset(dataset.columns):
            raise ValueError("dataset must have 'text' and 'label' columns")
        if not {"original", "paraphrase"}.issubset(paraphrases.columns):
            raise ValueError("paraphrases must have 'original' and 'paraphrase' columns")
            
        # 2. Retain dataset assignment and filtering
        self.dataset = dataset
        self.paraphrases = paraphrases[paraphrases["original"].isin(dataset["text"])].copy()
        
        # We explicitly skip calling super().__init__() to avoid 
        # the model instantiation and embedding loop in the base class.

    def select(self, n_samples: int, target_label: int) -> pd.DataFrame:
            """
            Selects paraphrases aiming for a total of n_samples.
            
            Args:
                n_samples (int): Total number of augmented samples needed.
                target_label (int): The minority class label. Included primarily 
                                    to maintain signature consistency with other selectors.
            """
            # 1. Count the number of unique original minority sentences
            # (Relying on the pipeline guarantee that paraphrases only contains the target class)
            num_originals = self.paraphrases["original"].nunique()
            
            if num_originals == 0:
                return pd.DataFrame(columns=self.paraphrases.columns)
                
            # 2. Calculate k samples per original sentence
            k_samples_per_original = max(1, n_samples // num_originals)

            # 3. Group and select
            selected_paraphrases = (
                self.paraphrases
                .groupby("original")
                .head(k_samples_per_original)
                .reset_index(drop=True)
            )
            
            return selected_paraphrases

class EntropySelector(Selector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Internal classifier to estimate uncertainty
        self.prober = LogisticRegression(max_iter=1000)
        self._fit_prober()

    def _fit_prober(self):
        """Trains a quick linear head on existing embeddings."""
        X = self.original_embeddings
        y = self.dataset["label"].values
        self.prober.fit(X, y)

    def select(self, n_samples: int, target_label:int):
        """Selects the most 'confusing' paraphrases based on entropy."""
        candidates = self.paraphrases["paraphrase"].tolist()
        cand_embs = self._embed(candidates)

        # 1. Get probability distributions for each candidate
        # probs shape: (n_candidates, n_classes)
        probs = self.prober.predict_proba(cand_embs)

        # 2. Calculate Shannon Entropy for each row
        # base=2 gives bits, but the ranking remains the same regardless of base
        cand_entropy = entropy(probs, axis=1)

        # 3. Rank by highest entropy (most uncertain)
        self.paraphrases["entropy"] = cand_entropy
        return self.paraphrases.nlargest(n_samples, "entropy")

class ManifoldCoverageSelector(Selector):
    def select(self, n_samples: int, target_label: int, quality_threshold: float = 0.7):
        """
        Selects samples that maximize coverage of the minority manifold 
        while staying within a similarity threshold of the class prototype.
        """
        # 1. Calculate the Prototype (Mean Vector) for the target class
        target_texts = self.dataset[self.dataset["label"] == target_label]["text"]
        target_embs = np.array([self.dataset_embeddings[t] for t in target_texts])
        prototype = target_embs.mean(axis=0).reshape(1, -1)

        # 2. Embed and filter by 'Prototype Alignment' (Quality Control)
        # We don't want to pick a diverse set of outliers; we want a diverse set of valid samples.
        candidates = self.paraphrases["paraphrase"].tolist()
        cand_embs = self._embed(candidates)
        
        proto_sims = cosine_similarity(cand_embs, prototype).flatten()
        
        # Keep only candidates that are semantically aligned with the class
        valid_indices = np.where(proto_sims >= quality_threshold)[0]
        if len(valid_indices) < n_samples:
            # Fallback if threshold is too strict
            valid_indices = np.argsort(proto_sims)[-max(n_samples, len(proto_sims)//2):]
            
        filtered_embs = cand_embs[valid_indices]
        filtered_df = self.paraphrases.iloc[valid_indices].copy()

        # 3. Greedy Selection for Maximum Coverage
        # We pick the first sample closest to the prototype, 
        # then iteratively pick samples furthest from our already selected set.
        selected_indices = [np.argmax(proto_sims[valid_indices])]
        remaining_indices = list(set(range(len(filtered_embs))) - set(selected_indices))
        
        for _ in tqdm(range(1, n_samples), desc="Maximizing Coverage"):
            if not remaining_indices:
                break
                
            # Compute similarity between remaining candidates and already selected ones
            sim_matrix = cosine_similarity(filtered_embs[remaining_indices], filtered_embs[selected_indices])
            
            # For each candidate, find its MAXIMUM similarity to the already selected set.
            # We want to pick the candidate whose 'max similarity' is the LOWEST.
            # (i.e., the most novel/distant sample)
            max_sims = sim_matrix.max(axis=1)
            best_idx_in_remaining = np.argmin(max_sims)
            
            selected_indices.append(remaining_indices.pop(best_idx_in_remaining))
            
        return filtered_df.iloc[selected_indices]

class PrototypeSelector(Selector):
    def select(self, n_samples: int, target_label: int):
        # 1. Calculate the Prototype (Mean Vector) for the target class
        target_texts = self.dataset[self.dataset["label"] == target_label]["text"]
        target_embs = np.array([self.dataset_embeddings[t] for t in target_texts])
        prototype = target_embs.mean(axis=0).reshape(1, -1)

        # 2. Embed the candidate paraphrases
        candidates = self.paraphrases["paraphrase"].tolist()
        cand_embs = self._embed(candidates)

        # 3. Rank by Cosine Similarity to Prototype
        scores = cosine_similarity(cand_embs, prototype).flatten()
        self.paraphrases["score"] = scores
        
        return self.paraphrases.nlargest(n_samples, "score")

def select_augment_paraphrases(
    selector:str, 
    paraphrases:pd.DataFrame, 
    dataset:pd.DataFrame,
    num_samples:int,
    target_label:int,
    model:str="FacebookAI/xlm-roberta-base",
    )->pd.DataFrame:
    if selector=="prototype":
        selector:Selector=PrototypeSelector(
            dataset=dataset,
            paraphrases=paraphrases,
            model_name=model
        )
    elif selector=="manifold_coverage":
        selector:Selector=ManifoldCoverageSelector(
            dataset=dataset,
            paraphrases=paraphrases,
            model_name=model
        )
    elif selector=="entropy":
        selector:Selector=EntropySelector(
            dataset=dataset,
            paraphrases=paraphrases,
            model_name=model
        )
    elif selector=="first_k":
        selector:Selector=FirstKSelector(
            dataset=dataset,
            paraphrases=paraphrases,
            model_name=model
        )
    else:
        raise NotImplementedError(f"Selector {selector} not implemented of not supported")
    # Use selection on data
    augment = selector.select(num_samples, target_label)
    res = pd.DataFrame({
        "text":augment["paraphrase"],
        "original":augment["original"],
        "label":target_label
    })
    # TODO: Save res to selected examples
    # os.makedirs(absolute_path("output/selected_examples/").joinpath(args.result_prefix), exist_ok=True)
    # augment.to_csv(absolute_path("output/selected_examples/").joinpath(args.result_prefix).as_uri()+f"/fold_{fold_id}.csv")
    return res


if __name__ == "__main__":
    # ds = KNNScorer()
    # augment = pd.read_csv("output/selected_examples/january/llm_paraphrase-xlm-roberta_1_high/semeval2015/fold_0.csv")
    # dataset = load_dataset("semeval2024")
    # minority = dataset[dataset["label"]==minority_label(dataset)]
    # minority["original"]=minority["text"]
    # minority["paraphrase"]=minority["text"]
    # scored_aug = ds.score(
    #     dataset=dataset,
    #     candidates=minority,
    #     debug=True

    # )
    # scored_aug.to_csv("semeval2024test.csv")
    # print(scored_aug)
    # # print(scored_aug.describe())
    dataset = load_dataset("unbalanced/sst2")
    paraphrases = pd.read_csv("output/llm_paraphrasing/meta-llama/Llama-3.1-8B-Instruct/unbalanced_gen/unbalanced/sst2.csv")
    
    for selector in [
        "first_k"
        # "prototype",
        # "manifold_coverage",
        # "entropy"
    ]:
        selected = select_augment_paraphrases(
            selector=selector,
            paraphrases=paraphrases,
            dataset=dataset,
            num_samples=1,
            target_label=0,
        )
        print(selected.head(10))
