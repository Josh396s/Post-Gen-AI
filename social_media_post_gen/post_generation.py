from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import pandas as pd
import argparse
import evaluate
import torch
import re

# Model that will be used 
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

instructions = (
    "You are a social media expert for a financial services company. "
    "Your task: Write a short 2 sentence max post. The post must strictly focus only on the summary. "
    "Do not invent details, companies, or investment advice. "
    "Keep it professional and make sure the post doesn't contain any negative words. "
    "Example Post: Analysis of global bond markets suggests potential for diversifying portfolios. #Finance #MarketUpdate"
)

# Generates post given a summary 
def generate_post(model, tokenizer, summary, temp, top_p, max_new_tokens=80):
    prompt = instructions + f"Your Turn: Write a (1-2 sentence maximum) social media post for this summary:{summary}. Post:" 
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    generate_model = model.base_model if hasattr(model, "prompt_encoder") else model

    with torch.no_grad():
        output_ids = generate_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temp,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

def main(model_path, data_path, output_path, num_posts, temp, top_p, max_new_tokens, analyze):
    # Load data 
    data = pd.read_excel(data_path)
    
    # Initialize tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize base model 
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")

    # Load trained model 
    model = base_model if model_path == 'base' else PeftModel.from_pretrained(base_model, model_path)

    results = []

    # Load ROUGE metric if evaluate is set 
    if analyze:
        rouge = evaluate.load("rouge")
        article_best_scores = []
    else:
        rouge = None

    # Iterate through each entry and generate a post 
    for _, row in data.iterrows():
        post_id = row["postId"]
        summary = row["summary"]
        reference = row.get("postOriginal", None)

        print(f"\nGenerating posts for postId={post_id}...")
        article_scores = []

        # Generate multiple variations 
        for i in range(num_posts):
            post = generate_post(model, tokenizer, summary, temp, top_p, max_new_tokens)
            
            # Remove any unwanted links 
            cleaned = re.sub(r'http\S+', '', post) 
            entry = {"postId": post_id, "generatedPost": cleaned}

            # Compute ROUGE-L if reference is available and evaluate is set 
            if analyze and reference:
                rouge_output = rouge.compute(predictions=[cleaned], references=[reference], rouge_types=["rougeL"])
                score = rouge_output["rougeL"]
                entry["rougeL"] = score
                article_scores.append(score)

            results.append(entry)

    # Save results 
    df_out = pd.DataFrame(results)
    df_out.to_csv(output_path, index=False, encoding="utf-8-sig")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script that will generate post using specified trained model')
    parser.add_argument('model_path', type=str, help='Set to path of model. Use "base" for zero-shot')
    parser.add_argument('data_path', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('num_posts', type=int)
    parser.add_argument('temp', type=float)
    parser.add_argument('top_p', type=float)
    parser.add_argument('max_new_tokens', type=int)
    parser.add_argument('--analyze', action='store_true')
    
    args = parser.parse_args()
    main(args.model_path, args.data_path, args.output_path, args.num_posts, args.temp, args.top_p, args.max_new_tokens, args.analyze)