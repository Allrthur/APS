for aug in 0.5 "auto"; do
    for dataset in \
    sst2 \
    semeval2017 \
    imdb \
    rotten_tomatoes;
    do
        # SMOTE
        python classify.py --cross_validation --load_splits \
            --classifier LogisticRegression --model FacebookAI/xlm-roberta-base \
            --dataset unbalanced/$dataset --result_prefix "quickstart/smote/aug${aug}/$dataset" \
            --epochs 30 --tokenizer_max_len 128 --augment smote --aug_1toX $aug

        # EDA
        python classify.py --cross_validation --load_splits \
            --classifier LogisticRegression --model FacebookAI/xlm-roberta-base \
            --dataset unbalanced/$dataset --result_prefix "quickstart/eda/aug${aug}/$dataset" \
            --epochs 30 --tokenizer_max_len 128 --augment eda --aug_1toX $aug

        # PREDATOR
        python classify.py --cross_validation --load_splits \
            --classifier LogisticRegression --model FacebookAI/xlm-roberta-base \
            --dataset unbalanced/$dataset --result_prefix "quickstart/predator/aug${aug}/$dataset" \
            --epochs 30 --tokenizer_max_len 128 --augment cached_predator --aug_1toX $aug \
            --aug_path "output/predator_april_unbalanced_fix"

        # APS
        python classify.py --cross_validation --load_splits \
            --classifier LogisticRegression --model FacebookAI/xlm-roberta-base \
            --dataset unbalanced/$dataset --result_prefix "quickstart/paraphrase/llama+selectorentropy/aug_${aug}/$dataset" \
            --epochs 30 --tokenizer_max_len 128  --augment paraphrase \
            --aug_path output/llm_paraphrasing/meta-llama/Llama-3.1-8B-Instruct/unbalanced_gen/unbalanced/${dataset}.csv\
            --aug_selection "entropy" --aug_1toX $aug
    done
done