from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import PeftModel, get_peft_model, PromptTuningConfig, TaskType, PromptTuningInit
from sklearn.model_selection import train_test_split
from datasets import Dataset
import pandas as pd
import argparse
import torch

# Model that will be used
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# Reads in data from excel file
def load_data(file_path):
    return pd.read_excel(file_path)

# PEFT setup
def load_peft(num_vir_tokens):
    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.RANDOM,
        num_virtual_tokens=num_vir_tokens,
        tokenizer_name_or_path=MODEL_NAME,
    )
    return peft_config

# Tokenizes the training data for input
def tokenize_function(examples, tokenizer):

    inputs = [f"Write a social media post based on this summary:\n{x}" for x in examples['summary']]
    outputs = examples['postOriginal']

    # Concatenate input + output for causal LM
    concatenated = [inp + " " + out for inp, out in zip(inputs, outputs)]

    # Tokenize
    tokenized = tokenizer(
        concatenated,
        max_length=512,  # set max length for input+output
        truncation=True,
        padding="max_length"
    )

    # For causal LM, labels = input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# Hyperparameters for training
def load_trainer_args(path, learning_rate, epochs, batch_size):
    training_args = TrainingArguments(
        output_dir=path,  # Where the model predictions and checkpoints will be written
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,  # Higher learning rate than full Fine-Tuning
        num_train_epochs=epochs,
        logging_steps=50,
        save_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model='loss',
        fp16=True
    )
    return training_args

# Create Trainer object for training the model
def create_trainer(model, training_args, train_dataset, eval_dataset):
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    return trainer

def main(data_path, output_path, num_vir_tokens, num_epochs, lr, batch_size):
    # Load in full data
    data = load_data(data_path)

    # Split into train and validation
    train_data_df, val_data_df = train_test_split(data[['summary','postOriginal']], test_size=0.1, random_state=42)

    # Convert into HuggingFace datasets
    train_dataset = Dataset.from_pandas(train_data_df)
    val_dataset = Dataset.from_pandas(val_data_df)

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")

    # Initialize PEFT config
    peft_config = load_peft(num_vir_tokens)

    # Wrap model in PEFT
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    # Tokenize dataset
    tokenized_train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=train_dataset.column_names)
    tokenized_val_dataset = val_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=val_dataset.column_names)

    # Load training arguments
    training_args = load_trainer_args(output_path, lr, num_epochs, batch_size)
    
    # Load trainer
    trainer = create_trainer(model, training_args, tokenized_train_dataset, tokenized_val_dataset)

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script that utilizes prompt_tuning to finetune an LLM')
    parser.add_argument('data_path', type=str, help='Enter path for data that will be used')
    parser.add_argument('output_path', type=str, help='Enter path where model will be saved')
    parser.add_argument('num_vir_tokens', type=int, help='Enter number of virtual tokens to train')
    parser.add_argument('num_epochs', type=int, help='Enter number of epochs to train for')
    parser.add_argument('lr', type=float, help='Enter learning rate for training')
    parser.add_argument('batch_size', type=int, help='Input the batch size')
    
    args = parser.parse_args()

    main(args.data_path, args.output_path, args.num_vir_tokens, args.num_epochs, args.lr, args.batch_size)