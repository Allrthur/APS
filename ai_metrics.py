# Builtin Imports
import os
from typing import Union, List, Tuple, Dict
from pathlib import Path
import json
import statistics as st
import argparse
# Local Imports
import utils
# Installed Libs
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import numpy as np

import re


def parse_args()->argparse.Namespace:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--path", type=str, required=True)

def load_result_files(path:Union[Path, str])->List[pd.DataFrame]:
    """
    Loads pred files separately
    """
    path = utils.absolute_path(path)
    if path.is_file(): # If path is for a file, then load the file
        return [pd.read_csv(path)]
    else: # Else load each file and concat the DataFrame
        pathfiles = os.listdir(path)
        df = []
        for p in pathfiles:
            # # print(p)
            if p.startswith("preds"):
                df.append(pd.read_csv(path.joinpath(p)))
    return df

def save_metrics(path:Union[Path,str], metrics:pd.DataFrame, name:str="metrics.csv")->None:
    """
    Saves calculated metrics as a CSV
    """
    path = utils.absolute_path(path)
    if path.is_file(): path = path.parent
    path = path.joinpath(name)
    metrics.to_csv(path, index=False)

if __name__ == "__main__":
    # Load LogFiles
    result_paths = [
        # "results/dissertation-results/baseline/",
        "results/dissertation-results/eda/aug0.25/",
        "results/dissertation-results/eda/aug0.75/",
        "results/dissertation-results/predator/aug0.25/",
        "results/dissertation-results/predator/aug0.75/",
        # "results/dissertation-results/paraphrase/llama+selectorfirst_k/aug0.25/",
        # "results/dissertation-results/paraphrase/llama+selectorfirst_k/aug0.75/",
        # "results/dissertation-results/paraphrase/llama+random/aug0.25/seed1/",
        # "results/dissertation-results/paraphrase/llama+random/aug0.75/seed1/",
        # "results/dissertation-results/paraphrase/llama+random/aug0.25/seed2/",
        # "results/dissertation-results/paraphrase/llama+random/aug0.75/seed2/",
        # "results/dissertation-results/paraphrase/llama+random/aug0.25/seed3/",
        # "results/dissertation-results/paraphrase/llama+random/aug0.75/seed3/",
        # "results/dissertation-results/paraphrase/llama+random/aug0.25/seed4/",
        # "results/dissertation-results/paraphrase/llama+random/aug0.75/seed4/",
        # "results/dissertation-results/paraphrase/llama+random/aug0.25/seed5/",
        # "results/dissertation-results/paraphrase/llama+random/aug0.75/seed5/",
        # "results/dissertation-results/paraphrase/llama+selectorprototype/aug0.25/",
        # "results/dissertation-results/paraphrase/llama+selectorprototype/aug0.75/",
        "results/dissertation-results/paraphrase/llama+selectorentropy/aug0.25/",
        "results/dissertation-results/paraphrase/llama+selectorentropy/aug0.75/",
        # "results/dissertation-results-lr/baseline/",
        "results/dissertation-results-lr/eda/aug0.25/",
        "results/dissertation-results-lr/eda/aug0.75/",
        "results/dissertation-results-lr/predator/aug0.25/",
        "results/dissertation-results-lr/predator/aug0.75/",
        # "results/dissertation-results-lr/paraphrase/llama+selectorfirst_k/aug0.25/",
        # "results/dissertation-results-lr/paraphrase/llama+selectorfirst_k/aug0.75/",
        # "results/dissertation-results-lr/paraphrase/llama+random/aug0.25/seed1/",
        # "results/dissertation-results-lr/paraphrase/llama+random/aug0.75/seed1/",
        # "results/dissertation-results-lr/paraphrase/llama+random/aug0.25/seed2/",
        # "results/dissertation-results-lr/paraphrase/llama+random/aug0.75/seed2/",
        # "results/dissertation-results-lr/paraphrase/llama+random/aug0.25/seed3/",
        # "results/dissertation-results-lr/paraphrase/llama+random/aug0.75/seed3/",
        # "results/dissertation-results-lr/paraphrase/llama+random/aug0.25/seed4/",
        # "results/dissertation-results-lr/paraphrase/llama+random/aug0.75/seed4/",
        # "results/dissertation-results-lr/paraphrase/llama+random/aug0.25/seed5/",
        # "results/dissertation-results-lr/paraphrase/llama+random/aug0.75/seed5/",
        # "results/dissertation-results-lr/paraphrase/llama+selectorprototype/aug0.25/",
        # "results/dissertation-results-lr/paraphrase/llama+selectorprototype/aug0.75/",
        "results/dissertation-results-lr/paraphrase/llama+selectorentropy/aug0.25/",
        "results/dissertation-results-lr/paraphrase/llama+selectorentropy/aug0.75/",
    ]
    for result_path in result_paths:
        res = []
        datasets = [
            "sst2",
            "semeval2017",
            "imdb",
            "rotten_tomatoes",
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
            # "quora_insincere",
        ]
        for dataset in datasets:
            complete_path = result_path+dataset
            result_folds = load_result_files(complete_path)
            # List labels
            labels = sorted(result_folds[0]["label"].unique())
            # Create consolidated metrics for calculation of mean and stdev
            cv_p = {label:[] for label in labels}
            cv_r = {label:[] for label in labels}
            cv_f1 = {label:[] for label in labels}
            cv_cf_mtx = np.zeros(shape=(len(labels), len(labels)))

            # y_true = [1,0,0] #
            # y_pred = [1,1,1] #
            for idx, fold in enumerate(result_folds):
                
                p, r, f1,_ = precision_recall_fscore_support(
                    fold["label"], 
                    fold["pred"], 
                    labels=labels)
                cf_mtx = confusion_matrix(
                    fold["label"] ,
                    fold["pred"],
                    labels=labels
                )

                # print(f"Fold {idx}:")
                for idx, label in enumerate(labels):
                    # print(f"- Label: {label}")
                    # print(f" -- Prec.: {p[idx]}")
                    # print(f" -- Rec. : {r[idx]}")
                    # print(f" -- F1.  : {f1[idx]}")
                    cv_p[label].append(p[idx])
                    cv_r[label].append(r[idx])
                    cv_f1[label].append(f1[idx])
                cf_mtx_pd = pd.DataFrame(cf_mtx, columns=labels, index=labels)
                # print(f"Confusion Matrix [X=preds, Y=true]: \n{cf_mtx_pd}")
                cv_cf_mtx = cv_cf_mtx+cf_mtx
            # Prepare and save confusion matrix
            cv_cf_mtx = cv_cf_mtx/len(result_folds)
            cfd = ConfusionMatrixDisplay(confusion_matrix=cv_cf_mtx, display_labels=labels)
            cfd.plot()
            figpath = utils.absolute_path(complete_path)
            if figpath.is_file(): figpath = figpath.parent
            figpath = figpath.joinpath("cf_mtx.png")
            plt.savefig(figpath)

            if len(result_folds)>1:

                # for label in labels:
                #     # print(f"Mean (Stdev.) for label {label}:")
                #     # print(f"- Prec.: {st.mean(cv_p[label])}({st.stdev(cv_p[label])})")
                #     # print(f"- Rec. : {st.mean(cv_r[label])}({st.stdev(cv_r[label])})")
                #     # print(f"- F1.  : {st.mean(cv_f1[label])}({st.stdev(cv_f1[label])})")
                cv_cf_mtx = cv_cf_mtx/len(result_folds)
                pd_cv_cf_mtx = pd.DataFrame(cv_cf_mtx, columns=labels, index=labels)
                # print(f"Confusion Matrix [X=preds, Y=true]:\n{pd_cv_cf_mtx}")
            
            # Save metrics to file
            metrics = []
            for label in labels:
                label_metrics = [
                    {
                        "fold":i, 
                        "label":label,
                        "precision":cv_p[label][i], 
                        "recall":cv_r[label][i], 
                        "f1":cv_f1[label][i]
                    }

                    for i in range(len(result_folds)) 
                ]
                metrics.extend(label_metrics)
            metrics = pd.DataFrame(metrics).sort_values(by=["fold", "label"])
            save_metrics(complete_path, metrics, "metrics.csv")

            # calculate f1 mean and stdev for each label
            f1_mean = []
            f1_stdev = []
            # t_test = []
            labels = sorted(metrics["label"].unique())
            for label in labels:
                label_metrics:pd.DataFrame = metrics[metrics["label"]==label]
                f1_mean.append(label_metrics["f1"].mean())
                f1_stdev.append(label_metrics["f1"].std())
            res.append(pd.DataFrame({
                "dataset":[dataset]*len(labels),
                "label":labels,
                "f1_mean":f1_mean,
                "f1_stdev":f1_stdev
            }))
        resumed_metrics = pd.concat(res)    
        save_metrics(result_path, resumed_metrics, "resumed_metrics.csv")
        


    