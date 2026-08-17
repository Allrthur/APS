# Builtin
from typing import List
import re
import os
import time
import json
import argparse
import gc
# Installed Libs
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
import re
from collections import Counter
import tqdm
from jinja2.exceptions import TemplateError
# Local Imports
from load_dataset import load_dataset
from utils import label_count, minority_label, majority_label, autoselect_1toX
from utils import absolute_path

def parse_args()->argparse.Namespace:
    args = argparse.ArgumentParser()
    args.add_argument("--start", type=int, default=0)
    args.add_argument("--end", type=int, default=-1)
    args.add_argument("--dataset", type=str, default="")
    return args.parse_args()


class Paraphraser:
    def __init__(self, 
                 model_name="google/gemma-3-1b-it",
                 device="cpu",
                 start=None,
                 end=None,
        ):
        self.__tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.__model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map=device,
            torch_dtype=torch.bfloat16
        )
        self.__history = []
        self.__num_paraphrases = 1
        self.__restrictions:list[str]=[]
        self.__min_label = None
        self.__maj_label = None
        self.__gen_target = 0
        self.__gen_per_seed = 0
        self.__device = device
        self.__start = start
        self.__end = end

    def fit(self, dataset:pd.DataFrame, onetox=1):
        """
        Analyse the dataset and sets generation target as well as restrictions
        """
        # Get majority and minority labels
        self.__min_label = minority_label(dataset)
        self.__maj_label = majority_label(dataset)
        labelcount = label_count(dataset)
        self.__gen_target = labelcount[self.__maj_label] - labelcount[self.__min_label]
        self.__gen_per_seed = onetox
        input_size_mean, input_size_std = self.__calculate_input_size(dataset)
        self.__output_size = input_size_mean+input_size_std
        # maj_keywords, min_keywords = self.__keyword_analysis(dataset)
        # return maj_keywords, min_keywords
        # self.__restrictions.append(f"1. Use at least one of the following keywords: {min_keywords}\n")
        # self.__restrictions.append(f"2. Do not use the following keywords: {maj_keywords}\n")
        # Set restrictions
        system_message="""
### SYSTEM ROLE
You are a High-Fidelity Style Transfer & Data Augmentation Engine. Your goal is to generate N={N} paraphrases while aggressively preserving the specific "voice" of the input text.

### CRITICAL: STYLE MIRRORING
The most common error is "normalizing" text. You must NOT do this.
1. **Analyze the Input:** Look for capitalization patterns (e.g., all lowercase), punctuation habits (e.g., multiple !!!), slang, abbreviations (e.g., "u" vs "you"), and emojis.
2. **Mimic Strictly:** If the input is a messy Tweet, the output MUST be a messy Tweet. If the input is formal code documentation, the output MUST be formal.
3. **Do Not Correct:** Do not fix grammar, do not expand abbreviations, and do not formalize the tone.
4. **Language Anchor:** You must generate the output in the exact same language as the input. Even if the input is in {lang}, the output must remain in {lang}. Do not translate or localize.

### CRITICAL: OUTPUT FORMAT & TOKEN LIMITS
You are interacting with a parser that crashes on invalid JSON.
1. **Raw JSON Only:** Do not use markdown blocks (```json). Do not add introductions or conclusions.
2. **Atomic Batches:** Output a valid JSON object containing a list. If you feel you are approaching a token limit, close the JSON list and object early rather than cutting off in the middle of a string.
3. **Structure:**
{{
"paraphrases": [
    "paraphrase_1",
    "paraphrase_2"
]
}}

### FEW-SHOT EXAMPLES (Style Fidelity)

**Example 1: Informal/Tweet Style**
Input: "omg i cant believe he said that lol 💀"
N: 1
Result:
{{
"paraphrases": [
    "lol i am literally in shock that he said that 💀"
]
}}

**Example 2: Technical/Formal Style**
Input: "The function aborts if the return value is null."
N: 1
Result:
{{
"paraphrases": [
    "If a null value is returned, the procedure terminates."
]
}}
        """.format_map({
            "N": self.__gen_per_seed if self.__gen_per_seed<5 else 5,
            "lang":"Brazilian Portuguese" # WARNING: Hardcoded language
        })
        self.__history.append({"role":"system", "content":system_message})
        print(system_message)
    
    def __calculate_input_size(self, df:pd.DataFrame):
        token_counts = df['text'].apply(lambda x: len(self.__tokenizer.encode(str(x), padding=False)))
        mean_val = token_counts.mean()
        std_val = token_counts.std()
        # print("token counts:", mean_val, std_val)
        return (mean_val, std_val)
    
    def __generate(self, 
                   user_input, 
                   add_to_history=False,
                   temp=0.6
                   ):
        messages = self.__history + [{"role": "user", "content": user_input},
                                     {"role": "assistant", "content": "{"}]
        # print(user_input)
        text = self.__tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
            continue_final_message=True
        )
        inputs = self.__tokenizer(
            text, 
            return_tensors="pt"
        ).to(self.__model.device)
        response_ids = self.__model.generate(**inputs, 
                                             max_new_tokens=int(self.__output_size*1.5), 
                                             stop_strings=["\"]}"],
                                             tokenizer=self.__tokenizer,
                                             temperature=temp,
                                             pad_token_id=self.__tokenizer.eos_token_id
                                            )
        # Remove response from device
        response_ids.to("cpu")
        response_ids=response_ids[0][len(inputs.input_ids[0]):].tolist()
        response = self.__tokenizer.decode(response_ids, skip_special_tokens=True)
        # Update history
        if add_to_history:
            self.__history.append({"role": "user", "content": user_input})
            self.__history.append({"role": "assistant", "content": response})
        del inputs
        return "{"+response
    
    def __postproccess_response(self, res:str)->dict:
        if res=="{":return {"paraphrases":[]}
        if res.startswith("{{") and res.endswith("}}"):res=res[1:-1]
        try: res = json.loads(res)["paraphrases"]
        except: 
            print(f"\n{res}\n is a malformed JSON object ", end="")
            # with open("malformed_examples.txt", "a") as f:f.write(res+"\n")
            dummyres = res.replace("\n","").replace(" ", "")
            # Try to fix JSON output
            if dummyres.endswith("]"):
                fix_paraphrases = res+"}"
            elif dummyres.endswith(",]}"): # removing a trailing comma is complex
                s = res.rfind(",")
                fix_paraphrases = res[:s]+res[s+1:]
            elif dummyres.endswith("\""):
                fix_paraphrases = res+"]}"
            else: fix_paraphrases = res + "\"]}"
            # Try again to decode JSON
            try:
                res = json.loads(fix_paraphrases)["paraphrases"]
                print("but it was fixed successfuly.")
            except:
                res = []
                print("and it has been discarded.")
        return res
    
    def augment(self, dataset, save_path=None, high_temp=False)->pd.DataFrame:
        print(save_path)
        # check savepath for existing savefile
        try: 
            savedata = pd.read_csv(save_path)[["original","paraphrase"]]
            done_originals = savedata["original"].unique()
            last_sentence = done_originals[-1]
            curr_text_paraphrases = savedata[savedata["original"]==last_sentence]
            done_originals = set(done_originals[:-1])
            print(f"Data already exists in savepath, starting from {len(done_originals)}th sentence.")
            print(f"Just so you know, it is the following sentence:\n")
            print(last_sentence)
            print(f"And we have {len(curr_text_paraphrases)}/{self.__gen_per_seed} done for this sentence.")
        except FileNotFoundError as e:
            savedata=pd.DataFrame(columns=["original","paraphrase"])
            done_originals=set()
            curr_text_paraphrases = None
            print("Could not find saved data, starting from scratch.")
        originals = set(dataset[dataset["label"]==self.__min_label]["text"].iloc[self.__start:self.__end])
        minority_originals = list(originals-done_originals)
        for idx, text in enumerate(minority_originals):
            if curr_text_paraphrases is None: 
                curr_text_paraphrases = pd.DataFrame(columns=["original","paraphrase"])
            while len(curr_text_paraphrases) < self.__gen_per_seed:
                print(
                    f"\
Current sentence: {len(curr_text_paraphrases)}/{self.__gen_per_seed} ({(len(curr_text_paraphrases)/self.__gen_per_seed)*100:.2f}%)\n\
 Total sentences: {idx}/{len(minority_originals)} ({(idx/len(minority_originals))*100:.2f}) \
                    "
                    )
                try:paraphrases = self.__generate(text, False)
                except torch.OutOfMemoryError as e:
                    paraphrases = "{\"paraphrases\":\"[]\"}"
                    print("Out of Memory detected, breaking out of loop")
                    break
                paraphrases = self.__postproccess_response(paraphrases)
                curr_text_paraphrases=pd.concat([
                    curr_text_paraphrases, 
                    pd.DataFrame({
                        "original":text,
                        "paraphrase":paraphrases
                    }).drop_duplicates(subset=["paraphrase"])
                ])
                savedata = pd.concat([savedata, curr_text_paraphrases]).drop_duplicates(subset="paraphrase")
                if save_path: 
                    os.makedirs(absolute_path(save_path).parent, exist_ok=True)
                    savedata[["original","paraphrase"]].to_csv(save_path)
            curr_text_paraphrases = None
        return savedata
        
# Example Usage
if __name__ == "__main__":
    
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    OUTPUT_PATH = f"output/llm_paraphrasing/{MODEL_NAME}/toy-outputs/"
    DEVICE = "auto"

    os.makedirs(absolute_path(OUTPUT_PATH), exist_ok=True)
    args = parse_args()
    
    # if all arguments are default, then run the script as normal
    if (args.start == 0 and
        args.end == -1 and
        args.dataset == ""):
        
        datasets = [
            "unbalanced/sst2",
            "unbalanced/semeval2017",
            "unbalanced/imdb",
            "unbalanced/rotten_tomatoes"
        ]

        for dataset_name in datasets:
            dataset = load_dataset(dataset_name) # .sample(n=5,random_state=1)
            p = Paraphraser(
                model_name=MODEL_NAME,
                device=DEVICE
            )
            # input("Press any key to continue")
            onetox=((autoselect_1toX(dataset)+1)*3)
            p.fit(dataset, onetox=onetox)
            time_start = time.time()
            # print("Going into augment: ", len(dataset))
            # input("Press any key to continue")
            res = p.augment(
                    dataset=dataset, 
                    save_path=absolute_path(f"{OUTPUT_PATH}{dataset_name}.csv"),
                    high_temp=True
                )
            time_end = time.time()
            res.to_csv(absolute_path(f"{OUTPUT_PATH}{dataset_name}.csv"))
            del p
    else:
        print("Detected non default arguments")
        print(args)
        dataset_name = args.dataset
        dataset = load_dataset(dataset_name)
        p = Paraphraser(
            model_name=MODEL_NAME,
            device=DEVICE,
            start=args.start,
            end=args.end
        )
        onetox=((autoselect_1toX(dataset)+1)*3)
        p.fit(dataset, onetox=onetox)
        res = p.augment(
                dataset=dataset, 
                save_path=absolute_path(f"{OUTPUT_PATH}{dataset_name}.{args.start}to{args.end}.csv"),
                high_temp=True,
            )
        time_end = time.time()
        res.to_csv(absolute_path(f"{OUTPUT_PATH}{dataset_name}.{args.start}to{args.end}.csv"))
