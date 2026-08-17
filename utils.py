from typing import Union, List

import pandas as pd
import os
import pathlib
import torch
import numpy as np
import random

def lock_seed(random_state:int)->None:
    # Lock seed
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    random.seed(random_state)

def absolute_path(path:str)->pathlib.Path:
    return pathlib.Path(__file__).parent.joinpath(path)

def balance_factor(df:pd.DataFrame)->dict:
    labels = list(df["label"].unique())
    major_label_count=len(max([df[df["label"]==label] for label in labels], key=len))
    bf = {label:len(df[df["label"]==label])/major_label_count for label in labels}
    return bf

def label_count(df:pd.DataFrame)->dict:
    """
    Counts labels in the dataset
    """
    
    labels = list(df["label"].unique())
    lc = {label:len(df[df["label"]==label]) for label in sorted(labels)}
    return lc

def minority_label(df:pd.DataFrame)->Union[str,int]:
    lc = label_count(df)
    if all(x == list(lc.values())[0] for x in lc.values()): return list(lc.keys())[0]
    return min(lc, key=lambda x:lc[x])

def majority_label(df:pd.DataFrame)->Union[str,int]:
    lc = label_count(df)
    if all(x == list(lc.values())[0] for x in lc.values()): return list(lc.keys())[1]
    return max(lc, key=lambda x:lc[x])

def labels_into_ids(df:pd.DataFrame)->List[int]:
    # TODO: add clause to not convert labels that are already ids
    labels = list(df["label"].unique())
    labels.sort()
    labels_to_id = {label:id for id, label in enumerate(labels)}
    print(labels_to_id)
    return [labels_to_id[label] for label in df["label"]]

def autoselect_1toX(df:pd.DataFrame, per:bool = False)->int:
    """
    Automatically selects 1toX based on how many sentence clones achieve parity (or closest)
    """
    lc = list(label_count(df).values())
    minor, major = min(lc), max(lc)
    return major-minor if per else (major//minor)-1 

if __name__ == "__main__":
    
    import load_dataset
    
    for dsname in [
        "imdb",
        "rotten_tomatoes",
        # "2Lsentistrength_youtube",
        # "2Lsentistrength_twitter",
        # "2Lsentistrength_digg",
        # "2Lsentistrength_myspace",
        # "2Lsentistrength_bbc",
        # "2Lvader_twitter",
        # "2Lvader_amazon",
        # "2Ldebate",
        # "2Ldigital_music",
        # "2Lenglish_dailabor",
        # "2Lluxury_beauty",
        # "twitter_topics",
        # "twitter_topics_4_0",
        # "twitter_topics_4_1",
        # "twitter_topics_4_5",
        # "twitter_topics_4_3",
        # "twitter_topics_2_0",
        # "twitter_topics_2_1",
        # "twitter_topics_2_5",
        # "twitter_topics_2_3",
        # Non-folded datasets
        # "amazon2023",
        # "amazon2023_extremes",
        # "amazon2023_5_1",
        # "amazon2023_5_2",
        # "dblp",
        # "dblp_9_8",
        # "dblp_9_6",
        # "dblp_9_2",
        # "mpqa"
        # "semeval2015",
        # "semeval2016",
        # "semeval2024"
        # "acm",
        # "yelp2013",
        # "yelp2013_extremes",
        # "reuters90",
        # "finphrasebank",
        # "semeval2025_eng_anger"     , # 87 / 12 
        # "semeval2025_eng_fear"      , # 41 / 58
        # "semeval2025_eng_joy"       , # 75 / 24 
        # "semeval2025_eng_sadness"   , # 68 / 31
        # "semeval2025_eng_surprise"  , # 69 / 30
        # "semeval2025_ptbr_anger"    , # 67 / 32
        # "semeval2025_ptbr_disgust"  , # 96 / 3
        # "semeval2025_ptbr_fear"     , # 95 / 4
        # "semeval2025_ptbr_joy"      , # 73 / 26
        # "semeval2025_ptbr_sadness"  , # 85 / 14
        # "semeval2025_ptbr_surprise" , # 93 / 6
        # "semeval2025_esp_anger"     , # 75 / 24
        # "semeval2025_esp_disgust"   , # 67 / 32
        # "semeval2025_esp_fear"      , # 84 / 15
        # "semeval2025_esp_joy"       , # 67 / 32
        # "semeval2025_esp_sadness"   , # 84 / 15
        # "semeval2025_esp_surprise"  , # 78 / 21
        # "semeval2025_deu_anger"     , # 70 / 29
        # "semeval2025_deu_disgust"   , # 68 / 31 
        # "semeval2025_deu_fear"      , # 90 / 9
        # "semeval2025_deu_joy"       , # 79 / 20
        # "semeval2025_deu_sadness"   , # 80 / 19
        # "semeval2025_deu_surprise"  , # 93 / 6
        # "semeval2025_eng_pn"        , # 76 / 23
        # "semeval2025_ptbr_pn"       , # 64 / 35
        # "semeval2025_esp_pn"        , # 68 / 31
        # "semeval2025_deu_pn"        , # 75 / 24
        # "semeval2025_eng_pnn"       , # 66 / 13 / 20
        # "semeval2025_ptbr_pnn"      , # 43 / 31 / 24
        # "semeval2025_esp_pnn"       , # 61 / 9 / 28
        # "semeval2025_deu_pnn"       , # 54 / 27 / 17
        # "finphrasebank",
        # "finphrasebank_posva",
        # "finphrasebank_neuva",
        # "finphrasebank_negva",
        # "semeval2025_eng_posva",
        # "semeval2025_eng_neuva",
        # "semeval2025_eng_negva",
        # "semeval2025_ptbr_posva",
        # "semeval2025_ptbr_neuva",
        # "semeval2025_ptbr_negva",
        # "semeval2025_esp_posva",
        # "semeval2025_esp_neuva",
        # "semeval2025_esp_negva",
        # "semeval2025_deu_posva",
        # "semeval2025_deu_neuva",
        # "semeval2025_deu_negva"
        # "twitter_topics_0_a",
        # "twitter_topics_1_a",
        # "twitter_topics_2_a",
        # "twitter_topics_3_a",
        # "twitter_topics_4_a",
        # "twitter_topics_5_a",
        # "dblp_0_a",
        # "dblp_1_a",
        # "dblp_2_a",
        # "dblp_3_a",
        # "dblp_4_a",
        # "dblp_5_a",
        # "dblp_6_a",
        # "dblp_7_a",
        # "dblp_8_a",
        # "dblp_9_a",
        # "quora_insincere"
    ]:    
        print(dsname)
        dataset = load_dataset.load_dataset(dsname)
        lc = label_count(dataset)
        for key in sorted(lc):
            print(f"    {key}", lc[key], int(100*(lc[key]/len(dataset))))
            # labels_into_ids(dataset)
        print("minority: ", minority_label(dataset))
        print("majority: ", majority_label(dataset))