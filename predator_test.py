#%%
# Built In
import os
from typing import Iterable, Union, Literal
import time
# Local
from load_dataset import load_dataset
from prep_dataset import augment_dataset, create_folds, split_train_test
from feature_extraction import extract_cls_token, domain_adapt_model
from lm_classifier import SequenceClassifierWrapper
from utils import labels_into_ids, absolute_path
from eda import eda_oversample
# Installed
from predator import Predator
import pandas as pd
from sklearn.model_selection import train_test_split

# EXPERIMENT_NAME = "predator_timebased_complement"
EXPERIMENT_NAME = "predator_time_test"

os.makedirs(f"results/{EXPERIMENT_NAME}", exist_ok=True)
# with open(f"results/{EXPERIMENT_NAME}/times.csv", mode="w") as f:
#     f.write("dataset,fold,train_time,gen_time\n")

def write_results(dataset, fold, train_time, gen_time):
    with open(f"results/{EXPERIMENT_NAME}/times.csv", mode="a") as f:
        f.write(f"{dataset},{fold},{train_time},{gen_time}\n")

def predator_samples_to_gen_calc(df_train, augment_ratio=1.0):
    import collections
    counter = collections.Counter(df_train["label"]).most_common()
    majority_class = counter[0][0]
    
    target_size = int(
        len(df_train[df_train["label"] == majority_class]) * augment_ratio
    )

    minority_class = counter[-1][0]
    minority_size = len(df_train[df_train["label"] == minority_class])

    classes_to_generate = set(df_train["label"].tolist())
    if augment_ratio == 1.0:
        classes_to_generate -= {majority_class}

    samples_to_create = target_size * len(classes_to_generate) - len(
        df_train[df_train["label"] != majority_class]
    )

    print(samples_to_create)

def time_predator(dataset):
    # Train classifier for each fold
    for fold_id in range(10):
        if fold_id <= 4 and dataset == "semeval2025_ptbr_surprise": 
            print("skipping fold: ", fold_id)
            continue
        # Unpack fold into train, val and test
        train_fold = pd.read_csv(f"dataset/saved_folds/{dataset}/train_fold{fold_id}.csv")
        val_fold = pd.read_csv(f"dataset/saved_folds/{dataset}/val_fold{fold_id}.csv")
        pr = Predator(
            train_fold, 
            val_fold, 
            device="cuda:0",
            num_majority_classes=1,
            generator_kwargs={
                # "model_name_or_path":"Qwen/Qwen3-0.6B"# f"arthurbittencourt/distillGPT2-{dataset}-fold{fold_id}"
                "device":"cuda:0"
            },
            filter_kwargs={
                # "model_name_or_path":"distillBERT"# f"arthurbittencourt/distillBERT-{dataset}-fold{fold_id}"
                "device":"cuda:0"
            },
            disable_filter=False
        )
        # print("Generator: ", pr.generator.model)
        # print("Filter: ", pr.filter.model)
        train_start_time = time.time()
        pr.train()
        train_end_time = time.time()
        # print(train_end_time - train_start_time)
        gen_start_time = time.time()
        aug = pr.augment(
            max_iterations=100,
            iterate_until_seconds=3600*3
        )
        gen_end_time = time.time()
        # pr.generator.model.push_to_hub(repo_id=f"distillGPT2-{dataset}-fold{fold_id}", private=False)
        # pr.generator.tokenizer.push_to_hub(repo_id=f"distillGPT2-{dataset}-fold{fold_id}", private=False)
        # pr.filter.model.push_to_hub(repo_id=f"distillBERT-{dataset}-fold{fold_id}", private=False)
        # pr.filter.tokenizer.push_to_hub(repo_id=f"distillBERT-{dataset}-fold{fold_id}", private=False)
        write_results(dataset, fold_id, train_end_time - train_start_time, gen_end_time - gen_start_time)
        os.makedirs(f"output/{EXPERIMENT_NAME}/{dataset}", exist_ok=True)
        aug.to_csv(f"output/{EXPERIMENT_NAME}/{dataset}/fold{fold_id}.csv")
        
    
if __name__ == "__main__":
    datasets = [
        # "unbalanced/sst2",
        # "unbalanced/semeval2017",
        # "unbalanced/imdb",
        # "unbalanced/rotten_tomatoes",
        "semeval2015",
        "semeval2024",
        "semeval2025_eng_anger",
        # "semeval2025_eng_joy",
        # "semeval2025_ptbr_disgust",
        # "semeval2025_ptbr_sadness",
        # "semeval2025_ptbr_surprise",
        # "semeval2025_esp_surprise",
        # "semeval2025_deu_sadness",
        # "semeval2025_deu_surprise",
        "twitter_topics_0_a",
        "twitter_topics_1_a",
        # "dblp_2_a",
        # "dblp_6_a",
        # "quora_insincere"
    ]
    # print(len(datasets))
    # exit()
    for dataset in datasets:
        print(dataset)
        time_predator(dataset)
# %%
