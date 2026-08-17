# %%
# Built-In
import argparse
import os
import datetime
import statistics as st
import random
from typing import Iterable, Union, Literal

# Installed
import wandb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Local (Assumindo que esses módulos existem no seu ambiente)
from load_dataset import load_dataset
from prep_dataset import augment_dataset, create_folds, split_train_test
from feature_extraction import extract_mean_pooled_embeddings, domain_adapt_model
from lm_classifier import SequenceClassifierWrapper
from utils import labels_into_ids, absolute_path, autoselect_1toX, minority_label, majority_label, label_count

# Constantes de Configuração
CLASSIC_ML_CLASSIFIERS = ["LogisticRegression", "MLP"]
CLASSIC_ML_AUGMENT = ["smote"]


# ===================================================================== #
#   FUNÇÕES DE SUPORTE E PARSER                                         #
# ===================================================================== #
def auto_int_or_float(value: str) -> Union[str, int, float]:
    normalized = str(value).strip().lower()
    if normalized == "auto": 
        return "auto"
    try:
        num_int = int(value)
        if num_int >= 1: 
            return num_int
    except ValueError: 
        pass
    try:
        num_float = float(value)
        if num_float < 1.0: 
            return num_float
    except ValueError: 
        pass
    raise argparse.ArgumentTypeError(f"Value must be 'auto', an int >= 1, or a float < 1.0. Got '{value}'")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-Validation Pipeline for Classic ML and LMs")
    
    # Dataset e Splits
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--load_splits", action="store_true")
    parser.add_argument("--save_used_splits", action="store_true")
    parser.add_argument("--undersample_min", type=float, default=1)
    parser.add_argument("--undersample_seed", type=int, default=1)
    parser.add_argument("--save_undersample_mode", action="store_true")
    parser.add_argument("--cross_validation", action="store_true")
    parser.add_argument("--partial_cv", action="store_true")
    
    # Augmentation
    parser.add_argument("--augment", type=str, default="")
    parser.add_argument("--aug_path", type=str, default="")
    parser.add_argument("--aug_selection", type=str, default="")
    parser.add_argument("--aug_scorer", type=str, default="bertscore")
    parser.add_argument("--aug_1toX", type=auto_int_or_float, default="auto")
    
    # Modelagem
    parser.add_argument("--classifier", type=str, required=True)
    parser.add_argument("--model", type=str, default="FacebookAI/xlm-roberta-base")
    parser.add_argument("--domain_adapt", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--tokenizer_max_len", type=int, default=120)
    parser.add_argument("--result_prefix", type=str, default="")
    parser.add_argument("--save_model", action="store_true")
    
    args = parser.parse_args()
    
    # Validações de Argumentos
    if args.classifier not in CLASSIC_ML_CLASSIFIERS and args.augment in CLASSIC_ML_AUGMENT:
        raise ValueError(f"O classificador {args.classifier} não suporta o aumento {args.augment}.")
    if args.load_splits and not args.cross_validation:
        raise ValueError("--load_splits é suportado apenas com --cross_validation.")
    if args.partial_cv and not args.cross_validation: 
        raise ValueError("--partial_cv não pode ser usado sem --cross_validation.")
    if args.augment in ["cached_predator", "paraphrase"] and not args.aug_path:
        raise ValueError(f"O aumento '{args.augment}' exige que '--aug_path' seja especificado.")
    
    args.original_model = args.model 
    print(f"Args Parseados: {args}")
    return args

def check_saved_preds(fold_num: int, path_prefix: str = "") -> bool:
    filename = f"results/{path_prefix}/preds_fold{fold_num}.csv"
    return os.path.exists(absolute_path(filename))

def save_preds(args, examples, preds, labels, fold_num: int, path_prefix: str = "", wandb_run=None):
    os.makedirs(absolute_path(f"results/{path_prefix}"), exist_ok=True)
    filename = f"results/{path_prefix}/preds_fold{fold_num}.csv"
    
    df_preds = pd.DataFrame({
        "text": examples,
        "pred": preds,
        "label": labels
    })
    df_preds.to_csv(absolute_path(filename), index=False, quoting=1) # quoting=1 garante aspas no texto
    
    if wandb_run:
        wandb_run.save(filename, base_path=absolute_path("results"))


# ===================================================================== #
#   MÓDULO DE AUMENTO DE DADOS                                          #
# ===================================================================== #
def get_aug_1toX(args, train_fold: pd.DataFrame) -> int:
    if args.aug_1toX == "auto": 
        return autoselect_1toX(train_fold)
    elif isinstance(args.aug_1toX, float) and args.aug_1toX < 1: 
        maj_size = len(train_fold[train_fold["label"] == majority_label(train_fold)])
        min_size = len(train_fold[train_fold["label"] == minority_label(train_fold)])
        return int(maj_size * args.aug_1toX / min_size) - 1
    return int(args.aug_1toX)

def apply_text_augmentation(args, train_fold: pd.DataFrame, val_fold: pd.DataFrame, fold_id: int) -> pd.DataFrame:
    """Aplica técnicas de aumento que geram TEXTO bruto (Data Augmentation pré-vetorização)."""
    if not args.augment or args.augment in CLASSIC_ML_AUGMENT:
        return train_fold
        
    aug_1toX = get_aug_1toX(args, train_fold)
    aug_df = None
    min_label = minority_label(train_fold)
    n_minority = len(train_fold[train_fold["label"] == min_label])
    num_samples = aug_1toX * n_minority
    
    print(f"Aplicando text augmentation: {args.augment} (Target samples: {num_samples})")
    
    if args.augment == "eda":
        from eda import eda_oversample
        aug_df = eda_oversample(train_fold, onetoX=aug_1toX, random_state=args.seed)
        
    elif args.augment == "predator":
        from predator import Predator
        pr = Predator(train_fold, val_fold, device="cuda:0", num_majority_classes=1)
        pr.train()
        aug_df = pr.augment(max_iterations=500)
        
    elif args.augment == "cached_predator":
        aug_df = pd.read_csv(absolute_path(f"{args.aug_path}/{args.dataset}/fold{fold_id}.csv"))
        aug_df = aug_df.head(num_samples)
        
    elif args.augment == "cached_paraphrase":
        print("READING PARAPHRASES FROM:")
        print(f"{args.aug_path}/{args.aug_1toX}/{args.dataset}/fold_{fold_id}.csv")
        sanitized_dataset_name = args.dataset.split("/")[-1] if args.dataset.startswith("unbalanced/") else args.dataset
        aug_df = pd.read_csv(f"{args.aug_path}aug{args.aug_1toX}/{sanitized_dataset_name}/fold_{fold_id}.csv")

    elif args.augment == "selected_paraphrase" or "paraphrase":
        from selector import select_augment_paraphrases
        paraphrases_df = pd.read_csv(absolute_path(args.aug_path))
        aug_df = select_augment_paraphrases(
            selector=args.aug_selection, paraphrases=paraphrases_df,
            dataset=train_fold, num_samples=num_samples, target_label=min_label
        )
    else:
        raise NotImplementedError(f"Augmentation method {args.augment} not implemented or supported")
        
    if aug_df is not None and not aug_df.empty:
        aug_savepath = absolute_path(f"output/selected_examples/{args.result_prefix}")
        os.makedirs(aug_savepath, exist_ok=True)
        aug_df.to_csv(f"{aug_savepath}/fold_{fold_id}.csv", index=False)
        
        aug_df = aug_df[["text", "label"]]
        print(f"Aumento finalizado. Gerados {len(aug_df)} novos exemplos.")
        return pd.concat([aug_df, train_fold]).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    
    return train_fold


# ===================================================================== #
#   PIPELINE 1: MACHINE LEARNING CLÁSSICO (TABULAR / EMBEDDINGS)        #
# ===================================================================== #
def run_classic_ml_pipeline(args, train_fold: pd.DataFrame, val_fold: pd.DataFrame, test_fold: pd.DataFrame, fold_id: int):
    print("--- Iniciando Classic ML Pipeline ---")
    
    # 1. Aumento de texto ANTES da extração (se não for SMOTE)
    train_fold = apply_text_augmentation(args, train_fold, val_fold, fold_id)
    
    # 2. Extração de representações numéricas
    is_text_augmented = args.augment not in CLASSIC_ML_AUGMENT+[""]
    print("Extraindo embeddings (Mean Pooling)...")
    X_train = extract_mean_pooled_embeddings(train_fold, model_name=args.model, dataset_name=args.dataset, fold_id=fold_id, fold_name="train", bypass_cache=is_text_augmented)
    X_val = extract_mean_pooled_embeddings(val_fold, model_name=args.model, dataset_name=args.dataset, fold_id=fold_id, fold_name="val")
    X_test = extract_mean_pooled_embeddings(test_fold, model_name=args.model, dataset_name=args.dataset, fold_id=fold_id, fold_name="test")
    
    y_train = train_fold["label"].values
    y_val = val_fold["label"].values
    
    X_train_np = np.stack(X_train)
    X_val_np = np.stack(X_val)
    X_test_np = np.stack(X_test)
    
    # 3. Aumento Numérico/Vetorial (SMOTE)
    if args.augment == "smote":
        print("Aplicando SMOTE aos embeddings extraídos...")
        from imblearn.over_sampling import SMOTE
        aug_1toX = get_aug_1toX(args, train_fold)
        
        class_counts = train_fold["label"].value_counts()
        minority_class = class_counts.idxmin()
        majority_class = class_counts.idxmax()
        
        target_count = int(class_counts[minority_class] * (1 + aug_1toX))
        target_count = min(target_count, class_counts[majority_class]) # Limita ao tamanho da classe majoritária
            
        sm = SMOTE(random_state=args.seed, sampling_strategy={minority_class: target_count})
        X_train_np, y_train = sm.fit_resample(X=X_train_np, y=y_train)
    
    # 4. Instanciação do Classificador
    clf_kwargs = {"random_state": args.seed, "max_iter": 1000, "verbose": 1}
    if args.classifier == "LogisticRegression":
        cl = LogisticRegression(**clf_kwargs)
    elif args.classifier == "MLP":
        cl = MLPClassifier(**clf_kwargs)
    else:
        raise NotImplementedError(f"Classificador '{args.classifier}' não implementado na pipeline clássica.")
        
    # 5. Treinamento (Fusão de Treino + Validação conforme regra original)
    print("Treinando o modelo clássico...")
    X_train_full = np.concatenate([X_train_np, X_val_np])
    y_train_full = np.concatenate([y_train, y_val])

    cl.fit(X=X_train_full, y=y_train_full)
    
    # 6. Predição
    print("Realizando predições no conjunto de teste...")
    return cl.predict(X=X_test_np)


# ===================================================================== #
#   PIPELINE 2: MODELOS DE LINGUAGEM (DEEP LEARNING / TEXTO BRUTO)      #
# ===================================================================== #
def run_language_model_pipeline(args, train_fold: pd.DataFrame, val_fold: pd.DataFrame, test_fold: pd.DataFrame, fold_id: int, labels: list, run):
    print("--- Iniciando Language Model Pipeline ---")
    
    # 1. Aplica aumento de texto
    train_fold = apply_text_augmentation(args, train_fold, val_fold, fold_id)
        
    # 2. Instancia o Wrapper
    freeze_lm = (args.classifier == "LMLR")
    trained_model_name = f"{args.result_prefix.replace('/', '-')}-fold{fold_id}"
    
    cl = SequenceClassifierWrapper(
        model_name=args.model, 
        random_state=args.seed, 
        freeze_lm=freeze_lm,
        num_labels=len(labels), 
        epochs=args.epochs,
        lr=args.lr,
        minimum_delta=0.001,
        patience=3,
        wandb_run=run,
        max_length=args.tokenizer_max_len,
        save_model=args.save_model,
        trained_model_name=trained_model_name,
    )
    
    # 3. Treinamento com Validação e Early Stopping
    print("Iniciando Fine-Tuning do Language Model...")
    cl.fit(
        X=train_fold["text"], 
        y=train_fold["label"].tolist(), 
        val_X=val_fold["text"],
        val_y=val_fold["label"].tolist()
    )
    
    # 4. Predição
    print("Realizando predições no conjunto de teste...")
    return cl.predict(X=test_fold["text"])


# ===================================================================== #
#   FLUXO PRINCIPAL E ORQUESTRADOR                                      #
# ===================================================================== #
def classify(args: argparse.Namespace):
    # 1. Preparação do Dataset Original
    ds = load_dataset(args.dataset)
    ds["label"] = labels_into_ids(ds)
    labels = sorted(ds["label"].unique())
    
    # 2. Undersampling Opcional
    if args.undersample_min != 1:
        ds_min = ds[ds["label"] == minority_label(ds)]
        ds_maj = ds[ds["label"] == majority_label(ds)]
        ds = pd.concat([
            ds_maj, 
            ds_min.sample(frac=args.undersample_min, random_state=args.undersample_seed)
        ]).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        
    if args.save_undersample_mode:
        save_dir = absolute_path(f"dataset/undersampled_datasets/{args.dataset}/")
        os.makedirs(save_dir, exist_ok=True)
        ds.to_csv(f"{save_dir}/dataset.csv", index=False)
        print("Undersampled dataset salvo. Encerrando execução.")
        return
        
    # 3. Geração de Folds / Splits
    if args.cross_validation:
        folds = create_folds(ds, random_state=args.seed)
    else:
        train_split, test_split = split_train_test(ds, random_state=args.seed)
        folds = [(train_split, test_split)]
        
    # 4. Iteração sobre os Folds
    for fold_id, fold in enumerate(folds):
        # Validação de pulo de fold (Partial CV ou Cache)
        if args.partial_cv and fold_id >= 9 - 8: # Manteve a lógica de "folds_to_skip = 8" -> if fold_id >= 1
            print(f"Partial CV ativado. Pulando fold {fold_id}...")
            continue
            
        if check_saved_preds(fold_id, args.result_prefix): 
            print(f"Resultados já existem para o fold {fold_id}. Pulando...")
            continue
            
        print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] Treinando e predizendo para o Fold {fold_id}")
        
        # Carregamento ou Geração dos Splits do Fold
        if args.load_splits:
            train_fold = pd.read_csv(absolute_path(f"dataset/saved_folds/{args.dataset}/train_fold{fold_id}.csv"))
            val_fold   = pd.read_csv(absolute_path(f"dataset/saved_folds/{args.dataset}/val_fold{fold_id}.csv"))
            test_fold  = pd.read_csv(absolute_path(f"dataset/saved_folds/{args.dataset}/test_fold{fold_id}.csv"))
        else:
            train_fold, test_fold = fold
            train_fold, val_fold = train_test_split(
                train_fold, test_size=0.1, random_state=args.seed, shuffle=True, stratify=train_fold["label"]
            )
            
        # Tratamento de Nulos
        for split_df in [train_fold, val_fold, test_fold]:
            split_df.fillna({"text": ""}, inplace=True)
        
        # Salvamento de Splits gerados (se solicitado)
        if args.save_used_splits:
            savepath = absolute_path(f"dataset/saved_folds/{args.dataset}/")
            os.makedirs(savepath, exist_ok=True)
            train_fold.to_csv(f"{savepath}train_fold{fold_id}.csv", index=False)
            val_fold.to_csv(f"{savepath}val_fold{fold_id}.csv", index=False)
            test_fold.to_csv(f"{savepath}test_fold{fold_id}.csv", index=False)
            print(f"Splits do fold {fold_id} salvos com sucesso.")
            continue

        # 5. Inicialização do Logger Unificado (WandB)
        run = wandb.init(entity="melll-uff", project="copia-mas-quao-igual", config=args, reinit=True)      
        
        # 6. Bifurcação Dinâmica de Execução
        if args.classifier in CLASSIC_ML_CLASSIFIERS:
            preds = run_classic_ml_pipeline(args, train_fold, val_fold, test_fold, fold_id)
        else:
            preds = run_language_model_pipeline(args, train_fold, val_fold, test_fold, fold_id, labels, run)
            
        # 7. Avaliação e Salvamento Unificado
        save_preds(
            args=args, examples=test_fold["text"], preds=preds, labels=test_fold["label"], 
            fold_num=fold_id, path_prefix=args.result_prefix, wandb_run=run
        )
        
        run.finish()
    
    print("\nExecução finalizada com sucesso!")

if __name__ == "__main__":
    args = parse_args()
    classify(args)
# %%