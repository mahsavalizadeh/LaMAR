import random
import torch.nn as nn
import torch
import json
import pandas as pd
import os
import sys
import transformers
import evaluate
from datasets import Dataset
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import ndcg_score

######################################################
# Set device
######################################################
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


######################################################
# Load data
######################################################
with open('train.json') as json_file:
    train_dict = json.load(json_file)

#with open('meta.json', 'r', encoding='utf-8') as f:
    #metas1 = json.load(f)

with open('meta_data_orig.json', 'r', encoding='utf-8') as f:
    full_meta_info = json.load(f)

with open('smap.json', 'r', encoding='utf-8') as f:
    smap = json.load(f)

full_meta_info = {smap.get(k, k): v for k, v in full_meta_info.items()}
meta_info  = {k: full_meta_info[k] for k in smap.values() if k in full_meta_info}
######################################################
# Data Preparation
######################################################

def prepare_data(train_dict, meta_info, max_seq_len=5, pool_len=50):
    #max_seq_len=5
    train_set = []
    valid_set = []
    examples = []

    ids = random.sample(range(0, len(meta_info)), pool_len)
    pool =""
    for i in ids:
        candidate = list(meta_info.keys())[i]
        pool += f"{candidate}: title is {meta_info[candidate]['title']},\n"

    for user_id, items in train_dict.items():
        #print(user_id, items)
        # We will build incremental subsequences

        if len(items)>= 6:
            for i in range(len(items)):

                if len(items) - i  >= 6:
                    history_items = items[i:max_seq_len+i]
                    tests = items[max_seq_len+i]


                    prompt_text = f"\nTask: Predict the next item ID and its title ONLY from **The candidate pool** based on user_{user_id} provided history. "
                    prompt_text += f"\n\nUser_{user_id} has a chronological order of interactions as follows:\n"
                    for h_it in history_items:
                        desc = meta_info.get(h_it, meta_info[h_it])
                        prompt_text += f" - {h_it}: {desc}\n"


                    prompt_text += f"\n**The candidate pool**:\n"

                    a = meta_info[tests]['title']
                    test_title_id = f"{tests}: title is {a}"
                    #print('test_title_id', test_title_id)
                    unshuffled_pool = pool
                    unshuffled_pool += test_title_id

                    items_in = unshuffled_pool.split(",\n")
                    random.shuffle(items_in)
                    shuffled_pool = ",\n".join(items_in)
                    #shuffled_pool = ",\n".join(f"{i+1}) {item}" for i, item in enumerate(items_in))
                    prompt_text += shuffled_pool

                    prompt_text += f"\n\Select only ONE item ID from **The candidate pool**. The choosed item ID and its title is:\n"
                    #f"\n\Return only ONE item ID along with its title, and choose it from **The candidate pool**. The choosed item ID and title is:\n"


                    target_text_test = test_title_id #tests
                    #print('prompt_text',prompt_text,'target_text_test', target_text_test)
                    examples.append((prompt_text, target_text_test))


        elif len(items)< 2:
            continue

        else:
            #print(len(items))
            history_items = items[:-1]
            #print(history_items)
            tests = items[-1]
            prompt_text = f"\nTask: Predict the next item ID and its title ONLY from **The candidate pool** based on user_{user_id} provided history. "
            prompt_text += f"\n\nUser_{user_id} has a chronological order of interactions as follows:\n"
            for h_it in history_items:
                desc = meta_info.get(h_it, meta_info[h_it])
                prompt_text += f" - {h_it}: {desc}\n"


            prompt_text += f"\n**The candidate pool**:\n"

            a = meta_info[tests]['title']
            test_title_id = f"{tests}: title is {a}"
            unshuffled_pool = pool
            unshuffled_pool += test_title_id

            items_in = unshuffled_pool.split(",\n")
            random.shuffle(items_in)
            shuffled_pool = ",\n".join(items_in)
            #shuffled_pool = ",\n".join(f"{i+1}) {item}" for i, item in enumerate(items_in))

            prompt_text += shuffled_pool

            prompt_text += f"\n\Select only ONE item ID from **The candidate pool**. The choosed item ID and its title is:\n"

            target_text_test = test_title_id #tests
            #print('prompt_text',prompt_text,'target_text_test', target_text_test)
            examples.append((prompt_text, target_text_test))
    return examples

examples = prepare_data(train_dict, meta_info, max_seq_len=5,pool_len=20)



######################################################
# Model and tokenizer setup
######################################################

HUGGINGFACE_ACCESS_TOKEN = "My_token"


base_model = "meta-llama/Llama-3.2-3B-Instruct"
print(f"Using base model: {base_model}")

tokenizer = AutoTokenizer.from_pretrained(
    base_model,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
    token=HUGGINGFACE_ACCESS_TOKEN
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto",
    token=HUGGINGFACE_ACCESS_TOKEN
)


if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    model.resize_token_embeddings(len(tokenizer))



######################################################
# Data processing functions
######################################################

def apply_chat_template(example):
    messages = [
        {"role": "system", "content": "You are a helpful recommendation system. When asked to predict the next item ID, respond ONLY with the item ID and its title from **The candidate pool**. Do not explain your reasoning or add any other text. DO NOT GENERATE NEW ITEM ID or TITLE, just select from **The candidate pool**. "},
        {"role": "user", "content": example['0']},
        {"role": "assistant", "content": example['1']}
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"prompt": prompt}

def tokenize_function(example):
    tokenized_inputs = tokenizer(
        example['prompt'],
        padding="max_length",
        truncation=True,
        max_length=1200, #1100 for 3 features, 1900 for 4 features
        return_tensors=None
    )
    tokenized_inputs['labels'] = [-100 if token == tokenizer.pad_token_id else token for token in tokenized_inputs['input_ids']]

    assert len(tokenized_inputs['input_ids']) > 0, "Empty input_ids"
    assert len(tokenized_inputs['attention_mask']) > 0, "Empty attention_mask"
    assert len(tokenized_inputs['labels']) > 0, "Empty labels"


    return tokenized_inputs



######################################################
# Prepare datasets
######################################################

data_for_llama = pd.DataFrame(examples)
ds = Dataset.from_pandas(data_for_llama)
new_dataset = ds.map(apply_chat_template)
new_datasets = new_dataset.select(range(6100))


train_dataset = new_datasets.shuffle().select(range(5000))
eval_dataset = new_datasets.shuffle().select(range(5000, 5100))
test_dataset = new_datasets.shuffle().select(range(5100, 6100))

"""
with open("test_dataset_4f.json", "r") as tests:
    test_dic = json.load(tests)
    
test_dataset = pd.DataFrame.from_dict(test_dic, orient='index')
test_dataset = Dataset.from_pandas(test_dataset)

with open("train_dataset_4f.json", "r") as trains:
    trains_dic = json.load(trains)
    
train_dataset = pd.DataFrame.from_dict(trains_dic, orient='index')
train_dataset = Dataset.from_pandas(train_dataset)

with open("eval_dataset_4f.json", "r") as evals:
    eval_dic = json.load(evals)
    
eval_dataset = pd.DataFrame.from_dict(eval_dic, orient='index')
eval_dataset = Dataset.from_pandas(eval_dataset)

"""
train_data = train_dataset.map(tokenize_function)
train_data = train_data.remove_columns(["0", "1", 'prompt'])




test_dataset.to_pandas().to_json('test_dataset.json', orient='index')
train_dataset.to_pandas().to_json('train_dataset.json', orient='index')
eval_dataset.to_pandas().to_json('eval_dataset.json', orient='index')

######################################################
# LoRA configuration
######################################################

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    base_model_name_or_path=base_model,
    target_modules=['q_proj', 'k_proj', 'v_proj']
)



######################################################
# Prepare model
######################################################

#model = model.to(device) #no need
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)




######################################################
# Data collator
######################################################

collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    return_tensors="pt",
    pad_to_multiple_of=8
)



######################################################
# Metrics computation
######################################################

def compute_recommendation_metrics(eval_dataset, model, tokenizer, k=10):
    model.eval()

    total_recall = []
    total_ndcg = []
    total_mrr = []
    #device = next(model.parameters()).device  # Get model's device #no need


    for idx,example in enumerate(eval_dataset):
        full_prompt = example['prompt']
        user_prompt = full_prompt.split("<|start_header_id|>assistant<|end_header_id|>")[0]
        #ground_truth = str(example['1'].split(':')[0])#.strip() #original
        ground_truth = str(example['1']).strip('.,! /;\&*%^#(')


        print(f"\nExample {idx + 1}:")
        print(f"Ground truth: {ground_truth}")
        #print("***************************user_prompt", user_prompt) #it is correct


        inputs = tokenizer(user_prompt, return_tensors="pt", truncation=True, padding=True) #padding="max_length",max_length=1900,) #750


        # Move inputs to the same device as model
        #inputs = {k: v.to(device) for k, v in inputs.items()} #no need
        try:
            with torch.no_grad():
                outputs = model.generate(
                    #input_ids=inputs.input_ids,
                    #attention_mask=inputs.attention_mask,
                    **inputs,
                    max_new_tokens=50,
                    num_return_sequences= k,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,

                )

            preds = [tokenizer.decode(out, skip_special_tokens=True).strip() for out in outputs]
            #print("*****************preds after decoding",preds, "********")


            #preds = [pred.split("assistant")[1].split("\n\n")[1].split(':')[0] for pred in preds] #original
            preds = [pred.split("assistant")[1].split("\n\n")[1].strip('.,! /;\&*%^#(') for pred in preds]
            print("*****************preds after spliting",preds, "********")




            #print("***********Raw predictions:***********")
            #for i, pred in enumerate(preds):
                #print(f"*******Pred {i+1}: {pred}")

            #break

            try:
                rank = preds.index(ground_truth) + 1  # rank is 1-indexed
                print(f"Found ground truth at rank {rank}")
            except ValueError:
                rank = None
                print("Ground truth not found in predictions")
                #print(f"Ground truth: {ground_truth}")
                #print("Predictions:")
                #for p in preds:
                    #print(f"- {p}")

            recall = 1 if rank is not None else 0  # Recall@10 is 1 if ground truth is found, 0 otherwise.
            ndcg = 1 / np.log2(rank + 1) if rank is not None else 0   # NDCG: if found, discounted gain is 1/log2(rank+1)
            mrr = 1 / rank if rank is not None else 0  # MRR: Reciprocal rank; 1/rank if found, else 0.

            total_recall.append(recall)
            total_ndcg.append(ndcg)
            total_mrr.append(mrr)

        except Exception as e:
            print(f"Error during generation: {e}")
            print(f"Input shape: {inputs['input_ids'].shape}")
            continue


    return {
        "recall@10": np.mean(total_recall),
        "ndcg@10": np.mean(total_ndcg),
        "mrr": np.mean(total_mrr)
    }


def compute_metrics(eval_preds):
    #global trainer

    metrics = compute_recommendation_metrics(trainer.eval_dataset, trainer.model, tokenizer, k=10)
    print("Evaluation metrics:", metrics)
    return metrics
"""

metric1 = evaluate.load("recall", "10")
#metric3 = evaluate.load("mrr")
def compute_metrics(eval_preds):

    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    print('********',preds, '********',preds[1])


    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    print(decoded_preds, '******decoded_preds*******')

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    print(decoded_labels, '******decoded_labels*******')

    recall_output = metric1.compute(predictions=decoded_preds, references=decoded_labels)["recall"]
    ndcg_output = ndcg_score(predictions=decoded_preds, references=decoded_labels)["ndcg"]
    mrr_output = compute_mrr(processed_preds, processed_refs)

    return {
        "recall@10": np.mean(recall_scores),
        "ndcg@10": np.mean(ndcg_scores),
        "mrr": np.mean(mrr_scores)
    }


    print("Evaluation metrics:",recall_output) #[recall_output,ndcg_output,mrr_output] )

    return {
        "recall@10": np.mean(recall_output),
        "ndcg@10": np.mean(ndcg_output),
        "mrr": np.mean(mrr_output)
    }
"""




######################################################
# Initialize trainer
######################################################

print("Using SFTTrainer...")
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=SFTConfig(  #In old version, I used transformers.TrainingArguments
        output_dir="./recommender_lora_model",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=20,
        warmup_ratio=0.1,
        num_train_epochs=30,
        learning_rate=2e-4,#2e-3
        #max_seq_length= 1500,
        fp16=True,
        logging_steps=10,
        optim='adamw_bnb_8bit',
        evaluation_strategy="epoch",#"steps",
        #eval_steps=10,
        save_strategy="epoch",#"steps",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="mrr",
        greater_is_better=True,
        lr_scheduler_type="cosine",
        gradient_checkpointing=False,
        max_grad_norm=1.0,
        logging_first_step=True,
        dataloader_num_workers=0,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="prompt",
    ),
    #data_collator=collator,
    compute_metrics=compute_metrics,
    #dataset_text_field="prompt",
    peft_config=lora_config,
    preprocess_logits_for_metrics=lambda logits, labels: torch.argmax(logits, dim=-1), # This allows us to reduce the size of the logits stored on the GPU and use a larger eval_accumulation_steps value
)


######################################################
#main
######################################################

if __name__ == "__main__":
    print("Starting training...")
    print(f"Using device: {device}")
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Evaluation dataset size: {len(eval_dataset)}")

    trainer.train()
    trainer.save_model("./recommender_lora_model")
    print("****Model was trained successfully****")

    # evaluate the model on the test set:
    print("Evaluating on test dataset...")
    test_results = compute_recommendation_metrics(test_dataset, trainer.model, tokenizer, k=10)
    print("Test Set Evaluation Metrics:", test_results)
