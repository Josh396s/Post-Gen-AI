from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

# Model that will be used 
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# Load data 
def load_data(path):
    return pd.read_excel(path)

# LoRA setup 
def load_lora(r, alpha, dropout):
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    return lora_config

# Tokenization function 
def tokenize_function(data, tokenizer):
    inputs = [f"Summary: {x}" for x in data['summary']]
    outputs = data['postOriginal']

    # Concatenate input + output 
    concatenated = [inp + " " + out for inp, out in zip(inputs, outputs)]

    # Create tokenizer arguments 
    tokenized = tokenizer(
        concatenated,
        padding="max_length",
        truncation=True,
        max_length=512
    )

    # Set tokenized labels 
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# Load trainer arguments 
def load_trainer_args(path, batch_size, lr, epochs):
    training_args = TrainingArguments(
        output_dir=path,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        num_train_epochs=epochs,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        fp16=True,
        gradient_accumulation_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss"
    )
    return training_args

# Load trainer 
def load_trainer(model, training_args, train_data, eval_data, tokenizer, data_collator):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    return trainer

def main(data_path, output_path, epochs, lr, batch_size, lora_r, lora_alpha, lora_dropout):
    # Load the data 
    data = load_data(data_path)

    # Create train/validation split 
    train_data_df, val_data_df = train_test_split(data[['summary','postOriginal']], test_size=0.1, random_state=42)

    # Huggingface Dataset 
    train_dataset = Dataset.from_pandas(train_data_df)
    val_dataset = Dataset.from_pandas(val_data_df)

    # Initialize tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize model 
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")

    # Initialize LoRA config 
    lora_config = load_lora(lora_r, lora_alpha, lora_dropout)

    # Wrap model in LoRA 
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # Tokenize dataset 
    tokenized_train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=train_dataset.column_names)
    tokenized_val_dataset = val_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=val_dataset.column_names)

    # Load training arguments 
    training_args = load_trainer_args(output_path, batch_size, lr, epochs)

    # Initialize collator for data batching 
    data_collator = DataCollatorForSeq2Seq(tokenizer, return_tensors="pt", padding=True)

    # Load trainer 
    trainer = load_trainer(model, training_args, tokenized_train_dataset, tokenized_val_dataset, tokenizer, data_collator)
 
    # Train model 
    trainer.train()

    # Save model 
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script that finetunes a model on summary/post pairs using LoRA for efficiency")
    parser.add_argument('data_path', type=str, help='Input path for training data')
    parser.add_argument('output_path', type=str, help='Set destination path to save trained model')
    parser.add_argument('epochs', type=int)
    parser.add_argument('lr', type=float)
    parser.add_argument('batch_size', type=int)
    parser.add_argument('lora_r', type=int)
    parser.add_argument('lora_alpha', type=float)
    parser.add_argument('lora_dropout', type=float)

    args = parser.parse_args()
    main(args.data_path, args.output_path, args.epochs, args.lr, args.batch_size, args.lora_r, args.lora_alpha, args.lora_dropout)