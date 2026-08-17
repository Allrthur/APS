# Builtin Imports
from typing import Literal, Union
from statistics import mean, stdev
import os
# Local Imports
import utils
from scorer import Scorer, CentroidDistanceScorer, CossineScorer
# Library Imports
from sklearn.model_selection import StratifiedKFold, train_test_split
import pandas as pd
from evaluate import load
from parascore import ParaScorer
import torch

def score_paraphrases(
        paraphrase:pd.DataFrame,
        dataset:pd.DataFrame=pd.DataFrame(columns=["text","label"]),
        scorer:str="bertscore",
        batch_size:int=6
    )->list[float]:
    # print(f"scoring using scorer:{scorer}")
    if scorer == "bertscore":
        bertscorer = load("bertscore")
        scores:list = bertscorer.compute(
            predictions=[str(e) for e in paraphrase["paraphrase"].to_list()],
            references=[str(e) for e in paraphrase["original"].to_list()],
            lang="en",
            rescale_with_baseline=True,
            batch_size=batch_size
        )["f1"]
    elif scorer == "cendtroidscorer":
        scorer:Scorer = CentroidDistanceScorer()
        scores = scorer.score(
            candidates=paraphrase,
            dataset=dataset,
        )
    elif scorer == "parascore":
        parascorer = ParaScorer(
            lang="en",
            model_type="bert-base-uncased"
        )
        _,_,f=parascorer.free_score(
            cands=[str(e) for e in paraphrase["paraphrase"].to_list()],
            sources=[str(e) for e in paraphrase["original"].to_list()],
            batch_size=batch_size
        )
        scores=f.tolist()
    elif scorer == "normparascore":
        parascorer = ParaScorer(
            lang="en",
            rescale_with_baseline=True
        )
        _,_,f=parascorer.free_score(
            cands=[str(e) for e in paraphrase["paraphrase"].to_list()],
            sources=[str(e) for e in paraphrase["original"].to_list()],
            batch_size=batch_size,
        )
        scores=f.tolist()
    elif scorer.startswith("parascore_"):
        scorer, alpha = scorer.split("_")
        parascorer = ParaScorer(
            lang="en",
            rescale_with_baseline=True,
        )
        _,_,f=parascorer.free_score(
            cands=[str(e) for e in paraphrase["paraphrase"].to_list()],
            sources=[str(e) for e in paraphrase["original"].to_list()],
            batch_size=batch_size,
            alpha=float(alpha),
        )
        scores=f.tolist()
    else:
        raise ValueError(f"scorer {scorer} not implemented or not supported")
    return scores

def augment_dataset(
        dataset:pd.DataFrame, 
        paraphrase:Union[pd.DataFrame, str], 
        selection:Literal["all","high","mid","low","random"],
        scorer:Literal["bertscore","parascore"]="bertscore",
        onetoX:Union[Literal["auto"],int]='auto',
        random_state:int=1,
        split:int=-1,
        savepath:str=None,
        return_augment:bool=False
    )->pd.DataFrame:
    if savepath:
        if os.path.exists(f"{savepath}/fold_{split}.csv"): 
            print(f"\n\nremoving pre-existing savefile {savepath}\n\n")
            os.remove(f"{savepath}/fold_{split}.csv")
    # Copy dataset
    res = dataset.copy()
    # If selection is random_X, separate selection and seed
    if selection.startswith("random_"):
        selection,random_state = selection.split("_")
        random_state=int(random_state)
    # If onetoX is auto, calculate how many paraphrases per original to achieve parity
    if onetoX == 'auto': onetoX = utils.autoselect_1toX(dataset)
    # Score paraphrases if needed
    if selection not in ["random", "all"]:
        paraphrase["score"]=score_paraphrases(paraphrase,dataset,scorer)
        paraphrase = paraphrase.sort_values(by="score", ascending=False)
    # Add highest lowest or middest
    original_texts = list(paraphrase["original"].unique())
    for original in original_texts:
        # Se o texto original não está no dataset, ele foi separado como teste
        if original not in dataset["text"].to_list():
            continue
        # Get elected paraphrases
        paraphrase_label = utils.minority_label(dataset)
        elected_paraphrases = paraphrase[paraphrase["original"]==original]
        assert type(elected_paraphrases)==pd.Series or pd.DataFrame
        # Verify onetoX
        if len(elected_paraphrases) < onetoX: 
            # raise ValueError(f"There are only {len(elected_paraphrases)} for {original}, cant select {onetoX} from that few.")
            continue
        # Select high, mid or low
        if selection=="high":
            elected_paraphrases=elected_paraphrases[:onetoX]
        elif selection=="mid":
            # raise NotImplementedError("value deprecated")
            start=(len(elected_paraphrases)-onetoX)//2
            end = start+onetoX
            elected_paraphrases=elected_paraphrases[start:end]
        elif selection=="low":
            elected_paraphrases=elected_paraphrases[-onetoX:]
        elif selection=="random":
            elected_paraphrases=elected_paraphrases.sample(n=onetoX, random_state=random_state)
        # Create a DataFrame to join with original samples
        elected_dataframe = pd.DataFrame({
            "text":elected_paraphrases["paraphrase"],
            "original":elected_paraphrases["original"],
            "label":[paraphrase_label]*len(elected_paraphrases)
        })
        # If dataset needs saving, save it
        if savepath: 
            os.makedirs(savepath, exist_ok=True)
            elected_paraphrases[["original","paraphrase"]].to_csv(
                f"{savepath}/fold_{split}.csv", mode="a", 
                header=not(os.path.exists(f"{savepath}/fold_{split}.csv")))
        # Add elected paraphrases to dataset
        res = pd.concat([res, elected_dataframe])

    # print(f"Added {len(res)-len(dataset)} examples to minority class.")
    
    # Return augmented dataset
    if return_augment: return elected_dataframe
    else:return res
        
def split_train_test(ds:pd.DataFrame, random_state:int=1)->list[pd.DataFrame]:
    train, test = train_test_split(ds, train_size=0.8, stratify=ds["label"], random_state=random_state)
    return train, test

def create_folds(dataframe:pd.DataFrame, n_splits=10, random_state:int=1)->list[pd.DataFrame]:
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    splits = skf.split(dataframe["text"], dataframe["label"])
    # Possible data leak?
    splits = [(dataframe.iloc[train], dataframe.iloc[test]) for train, test in splits]
    return splits


if __name__ == "__main__":
    from load_dataset import load_dataset
    from utils import label_count
    seed = 1
    datasets = [
        "semeval2015",
        "semeval2024",
    ]
    for dataset in datasets:
        df = load_dataset(dataset)
        folds = create_folds(df)
        print(f"=={dataset}==")
        # print(len(df))
        intersections = []
        for idx,fold in enumerate(folds):
            train_fold, test_fold = fold
            train_fold, val_fold = train_test_split(
                train_fold, 
                test_size=0.1, 
                random_state=seed, 
                shuffle=True, 
                stratify=train_fold["label"]
            )
            # print(len(train_fold), len(test_fold))
            def _aug_split(split, random_state):
                return augment_dataset(
                    dataset=split, 
                    paraphrase=pd.read_csv(f"output/llm_paraphrasing/meta-llama/Llama-3.1-8B-Instruct/iteration3d_full/{dataset}.fix.csv"), 
                    selection="high", 
                    scorer="parascore",
                    onetoX="auto",
                    random_state=random_state,  
                    return_augment=True
                )
            # print("augmenting...")
            aug = _aug_split(train_fold, seed)
            print(aug.head())



            
    

    

    