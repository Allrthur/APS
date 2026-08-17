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
from utils import label_count, minority_label, majority_label, autoselect_1toX
from utils import absolute_path
from feature_extraction import extract_cls_token
# Installed Libs
import torch
import pandas as pd
from transformers import pipeline
import numpy as np
from nltk.corpus import stopwords
import re
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.neighbors import NearestNeighbors
import nltk
from nltk.tag import pos_tag_sents
from nltk.tokenize import word_tokenize

class Scorer():
    def __init__(
        self,
        model_name:str
        ):
        self.model_name = model_name
    
    def fit(
        self,        
        )->None:
        pass
    
    def extract_fts(
            self,
            dataset:pd.DataFrame,
            candidates:pd.DataFrame
        )->tuple:
        # Separating majority and minority samples
        minority = dataset[dataset["label"]==minority_label(dataset)]
        majority = dataset[dataset["label"]==majority_label(dataset)]
        # Extract Features
        minority_fts = extract_cls_token(
            df=minority, 
            model_name=self.model_name
        )
        majority_fts = extract_cls_token(
            df=majority, 
            model_name=self.model_name
        )
        candidates["text"]=candidates["paraphrase"]
        candidates_fts = extract_cls_token(
            df=candidates,
            model_name=self.model_name
        )
        return minority_fts, majority_fts, candidates_fts

    def score(
        self,
        candidates:pd.DataFrame,
        dataset:pd.DataFrame,
        )->list:
        if ("paraphrase" and "original") not in candidates.columns:
            raise ValueError("candidates dataframe is expected to have a paraphrase and original columns")
        if ("text" and "label") not in dataset.columns:
            raise ValueError("dataset dataframe is expected to have a text and label columns")
    
    def select(
        self,
        )->pd.DataFrame:
        pass

class CossineScorer(Scorer):
    def __init__(
        self,
        model_name:str="FacebookAI/xlm-roberta-base"
        ):
        Scorer.__init__(
            self, 
            model_name=model_name
        )
    
    # WARNING: This function may not be scallable
    def score(
        self,
        candidates:pd.DataFrame,
        dataset:pd.DataFrame,
        alpha:float=0.5,
        beta:float=0.5
        )->pd.DataFrame:
        super().score(
            candidates,
            dataset
        )
        candidates = candidates.copy()
        # Extract Features
        minority_fts, majority_fts, candidates_fts = super().extract_fts(
            dataset=dataset,
            candidates=candidates,
        )
        # Create map functions
        def get_mean_similarity(candidate:torch.Tensor, refs:list[torch.Tensor]):
            return cosine_similarity([candidate]*len(refs), refs).mean()
        def get_stdev_similarity(candidate:torch.Tensor, refs:list[torch.Tensor]):
            return cosine_similarity([candidate]*len(refs), refs).std()
        def get_mean_distance(candidate:torch.Tensor, refs:list[torch.Tensor]):
            return cosine_distances([candidate]*len(refs), refs).mean()
        def get_stdev_distance(candidate:torch.Tensor, refs:list[torch.Tensor]):
            return cosine_distances([candidate]*len(refs), refs).std()
        # Classify each text with similarity to minority
        candidates["min_cossim_mn"]=[get_mean_similarity(cand, minority_fts) for cand in candidates_fts]
        candidates["min_cossim_st"]=[get_stdev_similarity(cand, minority_fts) for cand in candidates_fts]
        # Classify each text with distance to majority
        candidates["maj_cosdist_mn"]=[get_mean_distance(cand, majority_fts) for cand in candidates_fts]
        candidates["maj_cosdist_st"]=[get_stdev_distance(cand, majority_fts) for cand in candidates_fts]
        # # Debug only: Get cosine similarity from original to paraphrase
        # candidates["text"]=candidates["original"]
        # originals_fts = extract_cls_token(
        #     df=candidates,
        #     model_name=self.model_name
        # )
        # candidates["orig_cossim"]=[cosine_similarity(c_fts.reshape(1,-1),o_fts.reshape(1,-1)) for c_fts,o_fts in zip(candidates_fts,originals_fts)]
        # Save test csv
        candidates.drop(columns="text").to_csv("semeval2025test.csv")
        candidates["score"]=candidates["min_cossim_mn"]*alpha+candidates["maj_cosdist_mn"]*beta
        return candidates["score"].to_list()
    
class CentroidDistanceScorer(Scorer):
    def __init__(
        self,
        model_name:str="FacebookAI/xlm-roberta-base",
        cutoff:bool=False,
        ):
        Scorer.__init__(
            self, 
            model_name=model_name,
        )
        self.cutoff:bool=cutoff
    
    # WARNING: This function may not be scallable
    def score(
        self,
        candidates:pd.DataFrame,
        dataset:pd.DataFrame,
        alpha:float=1.0,
        beta:float=1.0,
        cutoff:bool=None
        )->pd.DataFrame:
        super().score(
            candidates,
            dataset
        )
        if not cutoff: cutoff = self.cutoff
        candidates = candidates.copy()
        # Extract Features
        minority_fts, majority_fts, candidates_fts = super().extract_fts(
            dataset=dataset,
            candidates=candidates,
        )
        # Create map functions
        def dist_to_centroid(candidate:torch.Tensor, refs:list[torch.Tensor]):
            refs:torch.Tensor = torch.stack(refs, dim=0)
            return torch.dist(candidate, refs.mean(dim=0)).tolist()
        # Classify each text with similarity to minority
        candidates["min_cent_dist"]=[dist_to_centroid(cand, minority_fts) for cand in candidates_fts]
        # Classify each text with distance to majority
        candidates["maj_cent_dist"]=[dist_to_centroid(cand, majority_fts) for cand in candidates_fts]
        # Aggregate score
        def aggregate_dist_score(dmin, dmaj) -> float:
            res = beta*dmaj-alpha*dmin 
            if cutoff: return res if res > 0 else 0
            else:      return res
        candidates["score"]=[aggregate_dist_score(dmin, dmaj) for dmin, dmaj in zip(candidates["min_cent_dist"],candidates["maj_cent_dist"])]
        return candidates["score"].to_list()
         
class POSScorer(Scorer):
    def __init__(
        self,
        model_name:str="FacebookAI/xlm-roberta-base"
        ):
        Scorer.__init__(
            self, 
            model_name=model_name
        )
        nltk.download('averaged_perceptron_tagger_eng')
    
    # WARNING: This function may not be scallable
    def score(
        self,
        candidates:pd.DataFrame,
        dataset:pd.DataFrame=None,
        lang:str="eng",
        alpha:float=1.0,
        beta:float=1.0
        )->pd.DataFrame:
        super().score(
            candidates,
            dataset
        )
        candidates = candidates.copy()
        original = [[pos for _,pos in sent] for sent in pos_tag_sents([word_tokenize(s) for s in candidates["original"].to_list()], lang="eng")]
        paraphra = [[pos for _,pos in sent] for sent in pos_tag_sents([word_tokenize(s) for s in candidates["paraphrase"].to_list()], lang="eng")]
        def posscore(para:list, orig:list):
            return sum(1 for x in para if x in orig)/len(orig)

        candidates["score"]=[posscore(p, o) for p, o in zip(paraphra,original)]
        return candidates["score"].to_list()

class KNNScorer(Scorer):
    def __init__(
        self,
        model_name:str="FacebookAI/xlm-roberta-base"
        ):
        Scorer.__init__(
            self, 
            model_name=model_name
        )
    
    # WARNING: This function may not be scallable
    def score(
        self,
        candidates:pd.DataFrame,
        dataset:pd.DataFrame,
        alpha:float=1.0,
        beta:float=1.0,
        debug:bool=False
        )->pd.DataFrame:
        # raise NotImplementedError("Scorer not implemented")
        super().score(
            candidates,
            dataset
        )
        if not debug: candidates = candidates.copy()
        # Extract Features
        minority_fts, majority_fts, candidates_fts = super().extract_fts(
            dataset=dataset,
            candidates=candidates,
        )
        dataset_fts = []
        dataset_fts.extend(minority_fts)
        dataset_fts.extend(majority_fts)
        min_last_pos = len(minority_fts)-1
        # print("min_last_pos", min_last_pos)
        # Init NNeighbors
        model = NearestNeighbors(n_neighbors=10)
        model.fit(dataset_fts)
        cand_neighs = model.kneighbors(candidates_fts, return_distance=False)
        # # Create map functions
        # def classify_neighborhood(cand_neigh):
        #     maj_neighs = sum([1 for neighbor in cand_neigh if neighbor > min_last_pos])
        #     min_neighs = sum([1 for neighbor in cand_neigh if neighbor <= min_last_pos])
        #     if maj_neighs > min_neighs: return "noise"
        #     elif min_neighs < maj_neighs: return "min"
        #     elif min_neighs == maj_neighs: return "danger"
        # # Classify each text within danger, minority or noise
        # candidates["neighbourhood"]=[classify_neighborhood(cand) for cand in cand_neighs]
        def __min_maj_neighs(cand_neigh, min):
            maj_neighs = sum([1 for neighbor in cand_neigh if neighbor > min_last_pos])
            min_neighs = sum([1 for neighbor in cand_neigh if neighbor <= min_last_pos])
            return min_neighs if min else maj_neighs
        candidates["min_neighs"]=[__min_maj_neighs(cand, True) for cand in cand_neighs]
        candidates["maj_neighs"]=[__min_maj_neighs(cand, False) for cand in cand_neighs]
        def __score(cand_neigh):
            maj_neighs = sum([1 for neighbor in cand_neigh if neighbor > min_last_pos])
            min_neighs = sum([1 for neighbor in cand_neigh if neighbor <= min_last_pos])
            if maj_neighs > min_neighs: return 0
            elif  maj_neighs < min_neighs: return (0.5 + maj_neighs/10)
            elif min_neighs == maj_neighs: return 1
            # return (maj_neighs, min_neighs)
        candidates["score"] = [__score(nclass) for nclass in cand_neighs]
        return candidates if debug else candidates["score"].to_list()
        # return [__score(nclass) for nclass in cand_neighs]

if __name__ == "__main__":
    ds = KNNScorer()
    augment = pd.read_csv("output/selected_examples/january/llm_paraphrase-xlm-roberta_1_high/semeval2015/fold_0.csv")
    dataset = load_dataset("semeval2024")
    minority = dataset[dataset["label"]==minority_label(dataset)]
    minority["original"]=minority["text"]
    minority["paraphrase"]=minority["text"]
    scored_aug = ds.score(
        dataset=dataset,
        candidates=minority,
        debug=True

    )
    scored_aug.to_csv("semeval2024test.csv")
    print(scored_aug)
    # print(scored_aug.describe())
