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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from sklearn.metrics import ndcg_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HUGGINGFACE_ACCESS_TOKEN = ""
base_model_path = "meta-llama/Llama-3.2-3B-Instruct"

with open("test_dataset_4f.json", "r") as tests:
    test_dic = json.load(tests)
    
test_dataset = pd.DataFrame.from_dict(test_dic, orient='index')
test_dataset = Dataset.from_pandas(test_dataset)
#print(type(test_dataset))
#print(test_dataset[0])

def run_inference_on_test_examples(test_dataset, new_model_path="./recommender_lora_model_4f"): #my fine-tuned model
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path,torch_dtype=torch.float16,device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path,
                                              padding_side="left",add_eos_token=True,add_bos_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = PeftModel.from_pretrained(base_model, new_model_path)

    model = model.merge_and_unload()
    test_results = compute_recommendation_metrics_test(test_dataset, model, tokenizer, k=10)
    return test_results
    


def compute_recommendation_metrics_test(eval_dataset, model, tokenizer, k=10):
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
                    max_new_tokens=60,
                    num_return_sequences= k,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    
                )
            
            preds = [tokenizer.decode(out, skip_special_tokens=True).strip() for out in outputs]
            #print("*****************preds after decoding",preds, "********")
            
        
            #preds = [pred.split("assistant")[1].split("\n\n")[1].split(':')[0] for pred in preds] #original
            preds = [pred.split("assistant")[1].split("\n\n")[1].strip('.,! /;\&*%^#(') for pred in preds]
            #print("*****************preds after spliting",preds, "********")
           
            

    
            try:
                rank = preds.index(ground_truth) + 1  # rank is 1-indexed
                #print(f"Found ground truth at rank {rank}")
            except ValueError:
                rank = None
                #print("Ground truth not found in predictions")
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

if __name__ == "__main__":
    print("*Inference time:*...")
    test_results = run_inference_on_test_examples(test_dataset, new_model_path="./recommender_lora_model_4f")
    print("Test Set Evaluation Metrics:", test_results)
   

