# builtin
import utils
from typing import List, Tuple, Literal
import json
# installed
import pandas as pd
import datasets as hds

def contradictory_dedupe(ds:pd.DataFrame, mode:Literal["drop", "mean"]="drop")->pd.DataFrame:
    if mode=="drop":
        contradictions = ds.groupby('text')['label'].transform('nunique') > 1
        return ds[~contradictions].copy()
    else:
        return ds.groupby('text')['label'].agg(lambda x: x.mode().iloc[0]).reset_index()

def load_unbalanced(dataset:str):
    return pd.read_csv(utils.absolute_path(f"dataset/undersampled_datasets/{dataset}/dataset.csv"))

def load_semeval2017()->pd.DataFrame:
    df = pd.read_csv(utils.absolute_path("dataset/semeval2017/SemEval2017-task4-dev.subtask-A.english.INPUT.txt"), sep="\t")
    def map_sentiments(sent:str):
        if sent=="positive": return 1
        elif sent=="negative": return 0
        else: return -1
    df["label"] = [map_sentiments(s) for s in df["sentiment"]]
    df = df[df["label"]!=-1]
    return df.drop_duplicates(subset="text")[["text","sentiment","label"]]

def load_ethos() -> pd.DataFrame:
    ds = pd.read_csv(utils.absolute_path('dataset/ethos/Ethos_Dataset_Binary.csv'), sep=";")
    ds["label"]=ds["isHate"].astype(int)
    ds["text"]=ds["comment"]
    ds = ds[["text", "label"]]
    ds = contradictory_dedupe(ds,"drop").drop_duplicates(subset="text")
    return ds

def load_amazon2014(category:int=1) -> pd.DataFrame:
    if category=='1':
        df = pd.read_json('dataset/amazon2014/reviews_Musical_Instruments_5.json.gz', lines=True)
    elif category=="2":
        df = pd.read_json('dataset/amazon2014/reviews_Patio_Lawn_and_Garden_5.json.gz', lines=True)
    elif category=='3':
        df = pd.read_json('dataset/amazon2014/reviews_Automotive_5.json.gz', lines=True)
    elif category=='4':
        df = pd.read_json('dataset/amazon2014/reviews_Amazon_Instant_Video_5.json.gz', lines=True)
    elif category=='5':
        df = pd.read_json('dataset/amazon2014/reviews_Office_Products_5.json.gz', lines=True)
    elif category=='6':
        df = pd.read_json('dataset/amazon2014/reviews_Digital_Music_5.json.gz', lines=True)
    else: df = pd.DataFrame()
    df = df[["reviewText", "overall"]]
    df["label"] = [(0 if e < 3 else 1) for e in df["overall"]]
    df["text"] = df["reviewText"]
    return df[["text", "label"]].drop_duplicates(subset="text")

def load_sst2()->pd.DataFrame:
    ds:pd.DataFrame = hds.load_dataset("stanfordnlp/sst2", split="train").to_pandas()\
        .rename(axis="columns", mapper={"sentence":"text"})
    ds = contradictory_dedupe(ds,"drop").drop_duplicates(subset="text")
    return ds

def load_cybertrolls()->pd.DataFrame:
    res = []
    with open(utils.absolute_path("dataset/cybertrolls/Dataset for Detection of Cyber-Trolls.json"), mode="r") as file:
        for line in file.readlines():
            l:dict = json.loads(line)
            res.append({"text":l["content"], "label":l["annotation"]["label"][0]})
    ds = pd.DataFrame(res)
    ds = contradictory_dedupe(ds,"drop").drop_duplicates(subset="text")
    return ds

def load_semeval2015()->pd.DataFrame:
    df = pd.read_csv(utils.absolute_path("dataset/semeval2015/SemEval15.csv"))
    df = df.rename(axis="columns", mapper={"tweet":"text", "class":"label"})
    return df

def load_semeval2016()->pd.DataFrame:
    df = pd.read_csv(utils.absolute_path("dataset/semeval2016/SemEval16.csv"))
    df = df.rename(axis="columns", mapper={"tweet":"text", "class":"label"})
    return df.drop_duplicates(subset="text")

def load_semeval2024()->pd.DataFrame:
    df = pd.read_json(utils.absolute_path("dataset/semeval2024/original_json/train.json"))[["text","labels"]]
    df["label"] = ["negative" if str(labelset)=="[]" else "positive" for labelset in df["labels"]]
    return df[["text","label"]]

# TODO: Finish semeval 2025 implementation
def load_semeval2025(classes:list=[])->pd.DataFrame:
    if len(classes)==1:
        lang = classes[0]
        return pd.read_csv(utils.absolute_path(f"dataset/semeval2025/track_a/train/{lang}.csv"))
    elif len(classes)==2:
        lang, emotion = classes
    else: raise ValueError("Loader expects two parameters for alias, such as semeval2025_language_emotion")
    df = pd.read_csv(utils.absolute_path(f"dataset/semeval2025/track_a/train/{lang}.csv"))
    # If needed, aglutinate classes between positive, neutral and negative
    if emotion in ["pn", "pnn", "posva", "neuva", "negva"]:
        labels = df[[col for col in  df.columns if col!="text" and col!="id"]].to_dict(orient="tight")
        def aglut_classes(labelist:list, hasdisgust:bool):    
            if hasdisgust:
                anger, disgust, fear, joy, sadness, surprise = labelist
            else:
                anger, fear, joy, sadness, surprise = labelist
                disgust = 0
            # If example has only joy or joy and surprise, it's positive
            if joy and not(anger or disgust or fear or sadness):
                return "positive"
            # If example has any negative emotion and no joy it's negative
            elif (not joy) and (anger or disgust or fear or sadness):
                return "negative"
            # If example has no emotion or only surprise, it's neutral
            elif not(joy or anger or disgust or fear or sadness):
                return "neutral"
            # Else, it has both positive and negative, eliminate
            else:
                return -1
        df["label"]=[aglut_classes(labelist, "disgust" in labels["columns"]) for labelist in labels["data"]]
        df = df[df["label"]!=-1]
        if emotion=="pn":
            df = df[df["label"]!="neutral"]
        elif emotion=="pnn": 
            pass
        elif emotion=="posva": 
            df["label"]=[1 if label=="positive" else 0 for label in df["label"]]
        elif emotion=="neuva":
            df["label"]=[1 if label=="neutral" else 0 for label in df["label"]]
        elif emotion=="negva":
            df["label"]=[1 if label=="negative" else 0 for label in df["label"]]
    else:
        df["label"] = df[emotion]
    return df[["text","label"]].drop_duplicates(subset="text")

def load_wassa2017()->pd.DataFrame:
    train = pd.read_csv(utils.absolute_path("dataset/wassa2017/emotion-labels-train.csv"))
    test = pd.read_csv(utils.absolute_path("dataset/wassa2017/emotion-labels-test.csv"))
    val = pd.read_csv(utils.absolute_path("dataset/wassa2017/emotion-labels-val.csv"))

    df = pd.concat([train, test, val])
    return df

def load_sentistrength_twitter()->pd.DataFrame:
    ds =[]
    for i in range(10):
        train_split = pd.read_parquet(utils.absolute_path(f"dataset/2Ldatasets/sentistrength_twitter_2L/train_fold_{i}.parquet"))
        test_split = pd.read_parquet(utils.absolute_path(f"dataset/2Ldatasets/sentistrength_twitter_2L/test_fold_{i}.parquet"))
        ds.append(train_split)
        ds.append(test_split)
    return pd.concat(ds)

def load_sentistrength_youtube()->pd.DataFrame:
    ds =[]
    for i in range(10):
        train_split = pd.read_parquet(utils.absolute_path(f"dataset/2Ldatasets/sentistrength_youtube_2L/train_fold_{i}.parquet"))
        test_split = pd.read_parquet(utils.absolute_path(f"dataset/2Ldatasets/sentistrength_youtube_2L/test_fold_{i}.parquet"))
        ds.append(train_split)
        ds.append(test_split)
    return pd.concat(ds)

def load_sentistrength_digg()->pd.DataFrame:
    ds =[]
    for i in range(10):
        train_split = pd.read_parquet(utils.absolute_path(f"dataset/2Ldatasets/sentistrength_digg_2L/train_fold_{i}.parquet"))
        test_split = pd.read_parquet(utils.absolute_path(f"dataset/2Ldatasets/sentistrength_digg_2L/test_fold_{i}.parquet"))
        ds.append(train_split)
        ds.append(test_split)
    return pd.concat(ds)

def load_sentistrength_myspace()->pd.DataFrame:
    ds =[]
    for i in range(10):
        train_split = pd.read_parquet(utils.absolute_path(f"dataset/2Ldatasets/sentistrength_myspace_2L/train_fold_{i}.parquet"))
        test_split = pd.read_parquet(utils.absolute_path(f"dataset/2Ldatasets/sentistrength_myspace_2L/test_fold_{i}.parquet"))
        ds.append(train_split)
        ds.append(test_split)
    return pd.concat(ds)

def load_sentistrength_bbc()->pd.DataFrame:
    ds =[]
    for i in range(10):
        train_split = pd.read_parquet(utils.absolute_path(f"dataset/2Ldatasets/sentistrength_bbc_2L/train_fold_{i}.parquet"))
        test_split = pd.read_parquet(utils.absolute_path(f"dataset/2Ldatasets/sentistrength_bbc_2L/test_fold_{i}.parquet"))
        ds.append(train_split)
        ds.append(test_split)
    return pd.concat(ds)

def load_vader_twitter()->pd.DataFrame:
    ds =[]
    for i in range(10):
        train_split = pd.read_parquet(utils.absolute_path(f"dataset/2Ldatasets/vader_twitter_2L/train_fold_{i}.parquet"))
        test_split = pd.read_parquet(utils.absolute_path(f"dataset/2Ldatasets/vader_twitter_2L/test_fold_{i}.parquet"))
        ds.append(train_split)
        ds.append(test_split)
    return pd.concat(ds)

def load_vader_amazon()->pd.DataFrame:
    ds =[]
    for i in range(10):
        train_split = pd.read_parquet(utils.absolute_path(f"dataset/2Ldatasets/vader_amazon_2L/train_fold_{i}.parquet"))
        test_split = pd.read_parquet(utils.absolute_path(f"dataset/2Ldatasets/vader_amazon_2L/test_fold_{i}.parquet"))
        ds.append(train_split)
        ds.append(test_split)
    return pd.concat(ds)
    
def load_debate()->pd.DataFrame:
    ds =[]
    for i in range(10):
        train_split = pd.read_parquet(utils.absolute_path(f"dataset/2Ldatasets/debate_2L/train_fold_{i}.parquet"))
        test_split = pd.read_parquet(utils.absolute_path(f"dataset/2Ldatasets/debate_2L/test_fold_{i}.parquet"))
        ds.append(train_split)
        ds.append(test_split)
    return pd.concat(ds)

def load_digital_music()->pd.DataFrame:
    ds=[]
    for i in range(5):
        train_split = pd.read_parquet(utils.absolute_path(f"dataset/2Ldatasets/digital_music_2L/train_fold_{i}.parquet"))
        test_split = pd.read_parquet(utils.absolute_path(f"dataset/2Ldatasets/digital_music_2L/test_fold_{i}.parquet"))
        ds.append(train_split)
        ds.append(test_split)
    return pd.concat(ds)

def load_english_dailabor()->pd.DataFrame:
    ds=[]
    for i in range(10):
        train_split = pd.read_parquet(utils.absolute_path(f"dataset/2Ldatasets/english_dailabor_2L/train_fold_{i}.parquet"))
        test_split = pd.read_parquet(utils.absolute_path(f"dataset/2Ldatasets/english_dailabor_2L/test_fold_{i}.parquet"))
        ds.append(train_split)
        ds.append(test_split)
    return pd.concat(ds)

def load_luxury_beauty()->pd.DataFrame:
    ds=[]
    for i in range(5):
        train_split = pd.read_parquet(utils.absolute_path(f"dataset/2Ldatasets/luxury_beauty_2L/train_fold_{i}.parquet"))
        test_split = pd.read_parquet(utils.absolute_path(f"dataset/2Ldatasets/luxury_beauty_2L/test_fold_{i}.parquet"))
        ds.append(train_split)
        ds.append(test_split)
    return pd.concat(ds)

def load_amazon2023_extremes(classes:List[int]|None=None)->pd.DataFrame:
    train_ds = hds.load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", split="full", trust_remote_code=True)
    train_ds = train_ds.select_columns(["text", "rating"]).to_pandas()
    train_ds = train_ds.rename(axis="columns", mapper={"rating":"label"})
    train_ds["label"]=[int(label) for label in train_ds["label"]]
    def relabel_rating(rating):
        if rating in [5,4]: return 1 # Majoritaria
        if rating in [2,1]: return 0 # Minoritaria
        else: return -1
    train_ds["label"]=[relabel_rating(label) for label in train_ds["label"]]
    train_ds = train_ds[train_ds["label"]!=-1]
    return train_ds.drop_duplicates(subset="text")

def load_amazon2023(classes:List[int]|None=None)->pd.DataFrame:
    train_ds = hds.load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", split="full", trust_remote_code=True)
    train_ds = train_ds.select_columns(["text", "rating"]).to_pandas()
    train_ds = train_ds.rename(axis="columns", mapper={"rating":"label"})
    train_ds["label"]=[int(label) for label in train_ds["label"]]
    def relabel_rating(rating):
        if classes:
            if rating == classes[0]: return 1 # Majoritaria
            if rating == classes[1]: return 0 # Minoritaria
            else: return -1
        else:
            return rating
    train_ds["label"]=[relabel_rating(label) for label in train_ds["label"]]
    if classes: train_ds = train_ds[train_ds["label"]!=-1]
    return train_ds.drop_duplicates(subset="text")

def load_dblp(classes:List[int]=None)->pd.DataFrame:
    ds = hds.load_dataset("waashk/dblp", split="all").to_pandas()
    def relabel(label):
        if classes:
            if classes[1]=="a":
                if label==classes[0]: return 1 # Classe de interesse presente
                else: return 0                 # Classe de interesse ausente
            else: # Separate two classes
                if label==classes[0]: return 1 # Majoritaria
                if label==classes[1]: return 0 # Minoritaria
                else: return -1
        else:
            return label
    ds["label"]=[relabel(label) for label in ds["label"]]
    if classes: ds = ds[ds["label"]!=-1]
    return ds.drop_duplicates(subset="text")

def load_mpqa()->pd.DataFrame:
    ds = hds.load_dataset("waashk/mpqa", split="test")
    return ds.to_pandas().drop_duplicates(subset="text")

def load_twitter_topics(classes:list=[])->pd.DataFrame:
    ds = hds.load_dataset("waashk/twitter", split="all").to_pandas()
    def relabel(label):
        if classes:
            if classes[1]=="a":
                if label==classes[0]: return 1 # Classe de interesse presente
                else: return 0                 # Classe de interesse ausente
            else: # Separate two classes
                if label==classes[0]: return 1 # Majoritaria
                if label==classes[1]: return 0 # Minoritaria
                else: return -1
        else:
            return label
    ds["label"]=[relabel(label) for label in ds["label"]]
    if classes: ds = ds[ds["label"]!=-1]
    return ds.drop_duplicates(subset="text")

def load_acm(classes:list=[])->pd.DataFrame:
    ds = hds.load_dataset("waashk/acm", split="test").to_pandas()
    def relabel(label):
        if classes:
            if label==classes[0]: return 1 # Majoritaria
            if label==classes[1]: return 0 # Minoritaria
            else: return -1
        else:
            return label
    ds["label"]=[relabel(label) for label in ds["label"]]
    if classes: ds = ds[ds["label"]!=-1]
    return ds.drop_duplicates(subset="text")

def load_yelp2013_extremes(classes:list=[])->pd.DataFrame:
    ds = hds.load_dataset("waashk/yelp_2013", split="test").to_pandas()
    def relabel_rating(rating):
        if rating in [5,4]: return 1 # Majoritaria
        if rating in [2,1]: return 0 # Minoritaria
        else: return -1
    ds["label"]=[relabel_rating(label) for label in ds["label"]]
    ds = ds[ds["label"]!=-1]
    return ds.drop_duplicates(subset="text")

def load_yelp2013(classes:list=[])->pd.DataFrame:
    ds = hds.load_dataset("waashk/yelp_2013", split="test").to_pandas()
    def relabel(label):
        if classes:
            if label==classes[0]: return 1 # Majoritaria
            if label==classes[1]: return 0 # Minoritaria
            else: return -1
        else:
            return label
    ds["label"]=[relabel(label) for label in ds["label"]]
    if classes: ds = ds[ds["label"]!=-1]
    return ds.drop_duplicates(subset="text")

def load_reuters90(classes:list=[])->pd.DataFrame:
    ds = hds.load_dataset("waashk/reut90", split="test").to_pandas()
    def relabel(label):
        if classes:
            if label==classes[0]: return 1 # Majoritaria
            if label==classes[1]: return 0 # Minoritaria
            else: return -1
        else:
            return label
    ds["label"]=[relabel(label) for label in ds["label"]]
    if classes: ds = ds[ds["label"]!=-1]
    return ds.drop_duplicates(subset="text")

def load_quora_insincere(classes:list=[])->pd.DataFrame:
    
    ds = pd.concat(
        [
            pd.read_csv(utils.absolute_path(f"dataset/quora_insincere/train_part{i}.csv"))[["question_text","target"]] \
                for i in range(3)
        ]
    )
    ds= ds.rename(
        axis="columns", 
        mapper={"question_text":"text", "target":"label"}, 
        errors="raise"
    )
    # print(utils.label_count(ds))
    return ds

def load_finphrasebank(classes:list=[])->pd.DataFrame:
    phrases = []
    labels = []
    with open(utils.absolute_path("dataset/Financial Phrasebank/Sentences_AllAgree.fix.txt"), mode="r") as file:
        for line in file.readlines():
            # separate text from label
            text, label = line.split("@")
            phrases.append(text)
            labels.append(label[:-1]) # Removing linebreak from class label
    df = pd.DataFrame({"text":phrases,"label":labels})
    if classes:
        if classes[-1]=="pn":
            df = df[df["label"]!="neutral"]
            df["label"]=[1 if label=="positive" else 0 for label in df["label"]]
        if classes[-1]=="posva":
            df["label"]=[1 if label=="positive" else 0 for label in df["label"]]
        if classes[-1]=="neuva":
            df["label"]=[1 if label=="neutral"  else 0 for label in df["label"]]
        if classes[-1]=="negva":
            df["label"]=[1 if label=="negative" else 0 for label in df["label"]]
    return df

def load_dataset(dataset:str):
    if dataset.startswith("unbalanced/"):
        dataset = dataset.split('/')[-1]
        return load_unbalanced(dataset)
    elif dataset == "sst2":
        return load_sst2()
    elif dataset == "ethos":
        return load_ethos()
    elif dataset == "cybertrolls":
        return load_cybertrolls()
    elif dataset.startswith("amazon2014"):
        dataset, category = dataset.split("_")
        return load_amazon2014(category=category)
    elif dataset == "semeval2017":
        return load_semeval2017()
    elif dataset   == "semeval2015":
        return load_semeval2015()
    elif dataset == "semeval2016":
        return load_semeval2016()
    elif dataset == "semeval2024":
        return load_semeval2024()
    # TODO: Register Semeval2025 as a dataset
    elif "semeval2025" in dataset:
        dataset_params = dataset.split("_")
        # print(dataset_params)
        if len(dataset_params)==2:
            return load_semeval2025([dataset_params[-1]])
        else: 
            return load_semeval2025(dataset.split("_")[-2:])
    elif dataset == "wassa2017":
        return load_wassa2017()
    elif dataset == "2Lsentistrength_twitter":
        return load_sentistrength_twitter()
    elif dataset == "2Lsentistrength_youtube":
        return load_sentistrength_youtube()
    elif dataset == "2Lsentistrength_digg":
        return load_sentistrength_digg()
    elif dataset == "2Lsentistrength_myspace":
        return load_sentistrength_myspace()
    elif dataset == "2Lsentistrength_bbc":
        return load_sentistrength_bbc()
    elif dataset == "2Lvader_twitter":
        return load_vader_twitter()
    elif dataset == "2Lvader_amazon":
        return load_vader_amazon()
    elif dataset == "2Ldebate":
        return load_debate()
    elif dataset == "2Ldigital_music":
        return load_digital_music()
    elif dataset == "2Lenglish_dailabor":
        return load_english_dailabor()
    elif dataset == "2Lluxury_beauty":
        return load_luxury_beauty()
    elif dataset == "mpqa":
        return load_mpqa()
    elif dataset == "quora_insincere":
        return load_quora_insincere()
    # Bases com filtragem de classes
    elif dataset == "finphrasebank":
        return load_finphrasebank()
    elif "finphrasebank" in dataset:
        return load_finphrasebank([i for i in dataset.split("_")])
    elif dataset == "amazon2023_extremes":
        return load_amazon2023_extremes()
    elif dataset == "amazon2023":
        return load_amazon2023()
    elif "amazon2023" in dataset:
        return load_amazon2023([int(i) if i!="a" else "a" for i in dataset.split("_")[-2:]])
    elif dataset == "dblp":
        return load_dblp()
    elif "dblp" in dataset:
        return load_dblp([int(i) if i!="a" else "a" for i in dataset.split("_")[-2:]])
    elif dataset == "twitter_topics":
        return load_twitter_topics()
    elif "twitter_topics" in dataset:
        return load_twitter_topics([int(i) if i!="a" else "a" for i in dataset.split("_")[-2:]])
    elif dataset == "acm":
        return load_acm()
    elif "twitter_acm" in dataset:
        return load_acm([int(i) if i!="a" else "a" for i in dataset.split("_")[-2:]])
    elif dataset == "yelp2013_extremes":
        return load_yelp2013_extremes()
    elif dataset == "yelp2013":
        return load_yelp2013()
    elif "yelp2013" in dataset:
        return load_yelp2013([int(i) if i!="a" else "a" for i in dataset.split("_")[-2:]])
    elif dataset == "reuters90":
        return load_reuters90()
    elif "reuters90" in dataset:
        return load_reuters90([int(i) if i!="a" else "a" for i in dataset.split("_")[-2:]])
    else:
        raise NotImplementedError(
            f"Loader function for {dataset} not implemented or not supported."
        )        

if __name__ == "__main__":
    datasets = [
        "unbalanced/sst2",
        "unbalanced/semeval2017",
        # "sst2",
        # "cybertrolls",
        # "amazon2014_1",
        # "amazon2014_2",
        # "amazon2014_3",
        # "amazon2014_4",
        # "amazon2014_5",
        # "amazon2014_6",
        # "semeval2017",
        # "ethos",
        # "finphrasebank_pn"
        # "semeval2015",
        # "semeval2024",
        # "semeval2025_eng_anger",
        # "semeval2025_eng_joy",
        # "semeval2025_ptbr_disgust",
        # "semeval2025_ptbr_sadness",
        # "semeval2025_ptbr_surprise",
        # "semeval2025_esp_surprise",
        # "semeval2025_deu_sadness",
        # "semeval2025_deu_surprise",
        # "twitter_topics_0_a",
        # "twitter_topics_1_a",
        # "dblp_2_a",
        # "dblp_6_a",
        # "quora_insincere"
    ]
    
    from utils import labels_into_ids

    for dataset in datasets:
        print(f"\n ==Testing {dataset} loader==")
        import os
        from prep_dataset import create_folds
        from sklearn.model_selection import train_test_split
        ds = load_dataset(dataset)
        ds["label"] = labels_into_ids(ds)
        print(ds.head())
        ds["label"] = labels_into_ids(ds)
        print("\n ==Label Count==")
        print(utils.label_count(ds))
        print("1toX: ", utils.autoselect_1toX(ds))
        print("\n ==Duplicates: ==")
        print(len(ds[ds.duplicated()]))
        print("\n ==Duplicates with both labels: ==")
        dupes = ds[ds.duplicated(subset="text")].sort_values(by="text")
        for sentence in dupes["text"].unique():
            if dupes[dupes["text"]==sentence]["label"].mean() != 0 and \
                dupes[dupes["text"]==sentence]["label"].mean() != 1:
                print(f"{sentence} has both labels")
