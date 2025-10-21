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
