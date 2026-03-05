from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import pandas as pd
import argparse
import evaluate
import torch
import sys
import re

# Model that will be used
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

instructions = (
    "You are a social media expert for a financial services company. "
    "Your task: Write a short 2 sentence max post. The post must strictly focus only on the summary. "
    "Do not invent details, companies, or investment advice. Do not mention random information unrelated "
    "to what was mentioned in the summary. Do not recommend or promote anything. "
    "Keep it professional and make sure the post doesn't contain any negative words "
    "Make it very similar to the posts you were fine-tuned on "
    "Example: "
    "Summary: The Congressional Budget Office estimates that next year U.S. government debt held by the public will exceed 100% of GDP. The CBO forecast assumes that all legislation will be enacted as it is currently written. If that doesn't happen, the numbers could go higher. U.S. interest payments on the debt are expected to surpass defense spending. Higher taxes might be required to meet debt service payments. Slower economic growth also could be expected, given that government spending would need to be re-routed to debt service. "
    "Post: Alarm bell? U.S. government debt is expected to rise sharply in the years ahead. Important disclosures: https://bit.ly/2JzEDWl "
)

# Generates post given a summary
def generate_post(model, tokenizer, summary, temp, top_p, max_new_tokens=80):
    prompt = instructions + f"Your Turn: Write a (1-2 sentence maximum) social media post for this summary:{summary}.  Post:" 
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    if hasattr(model, "prompt_encoder"):
        generate_model = model.base_model
    else:
        generate_model = model

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
    if model_path == 'base':
        model = base_model
    else:
        model =  PeftModel.from_pretrained(base_model, model_path)

    results = []

    # Load ROUGE metric if evaluate is set
    if analyze:
        rouge = evaluate.load("rouge")
        article_best_scores = []
    else:
        rouge = None
        article_best_scores = None

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

            # Compute ROUGE-L if reference is available and evaluate is set to True
            if analyze and reference:
                rouge_output = rouge.compute(
                    predictions=[cleaned],
                    references=[reference],
                    rouge_types=["rougeL"]
                )
                score = rouge_output["rougeL"]
                entry["rougeL"] = score
                article_scores.append(score)

            results.append(entry)
            print(f"  Variation {i+1}: {cleaned}")

        # Track best ROUGE score for current article
        if analyze and article_scores:
            best_score = max(article_scores)
            article_best_scores.append(best_score)
            print(f"\nBest ROUGE-L for postId={post_id}: {best_score:.3f}")

    # Average ROUGE-L across all articles
    if analyze and article_best_scores:
        avg_score = sum(article_best_scores) / len(article_best_scores)
        print(f"\nAverage ROUGE-L across all articles: {avg_score:.3f}")

    # Save results
    df_out = pd.DataFrame(results)
    df_out.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script that will generate post using specified trained model')
    parser.add_argument('model_path', type=str, help='Set to path of model that will be used. Use "base" if you would like to do zero-shot generation')
    parser.add_argument('data_path', type=str, help='Enter path of data')
    parser.add_argument('output_path', type=str, help='Enter path to save generated content')
    parser.add_argument('num_posts', type=int, help='Choose how many posts will be generated for each article')
    parser.add_argument('temp', type=float, help='Input the temperature used for generation')
    parser.add_argument('top_p', type=float, help='Input the top_p used for generation')
    parser.add_argument('max_new_tokens', type=int, help='Input the number of new tokens to generate')
    parser.add_argument('--analyze', action='store_true', help='Set if you want to analyze generated posts using ROUGE')
    
    args = parser.parse_args()

    main(args.model_path, args.data_path, args.output_path, args.num_posts, args.temp, args.top_p, args.max_new_tokens, args.analyze)