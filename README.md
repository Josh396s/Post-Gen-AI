# Financial Content Social Media Generator

An end-to-end machine learning pipeline designed to automate the creation of professional social media posts (LinkedIn, X, etc.) from financial articles. This project explores different LLM adaptation techniques to match a specific professional "voice" while maintaining factual accuracy.



## Project Overview
The pipeline transforms a raw URL into a series of social media post variations through four main stages:
1.  **Automated Scraping**: Uses Playwright to extract core article text while filtering out legal boilerplate, disclosures, and navigation links.
2.  **Summarization**: Implements a `BART-large-cnn` model to condense long-form financial articles into concise summaries.
3.  **Model Adaptation**: Compares three distinct approaches to generate high-quality posts:
    * **Zero-Shot/Few-Shot Prompting**: Direct instructions to the base model (Llama-3.2-1B-Instruct).
    * **Prompt-Tuning**: Training virtual tokens to guide style.
    * **LoRA (Low-Rank Adaptation)**: Efficiently fine-tuning model weights for specific tonal alignment.
4.  **Evaluation**: Uses ROUGE-L metrics to quantitatively measure semantic overlap between AI-generated posts and professional human-written examples.

## Key Findings
* **LoRA Performance**: LoRA fine-tuning was the most effective method, producing the most concise and stylistically accurate posts.
* **Metric Analysis**: While BLEU was found to be too strict for creative rephrasing, ROUGE-L proved to be a reliable metric for capturing semantic similarity.
* **Quantitative Results**:
    * **Base Model ROUGE-L**: ~0.092
    * **Prompt-Tuned ROUGE-L**: ~0.093
    * **LoRA Fine-Tuned ROUGE-L**: ~0.107

## Repository Structure
* `data_processing.py`: Handles web scraping, text cleaning, and BART-based summarization.
* `lora_tuning.py`: Script for fine-tuning Llama-3.2-1B-Instruct using LoRA.
* `prompt_tuning.py`: Script for training virtual tokens (Prompt-Tuning).
* `post_generation.py`: Generates social media variations and runs ROUGE evaluation.

## Usage
### 1. Preprocess Data
Scrape content and generate summaries from a list of URLs:
```bash
python data_processing.py input_data.xlsx
```

### 2. Training the Models
You can train the model using two different Parameter-Efficient Fine-Tuning (PEFT) methods:

* **LoRA Fine-Tuning**: Run the tuning script using the preprocessed data to fine-tune the base model with LoRA. 
  ```bash
  python lora_tuning.py data_path output_path epochs lr batch_size lora_r lora_alpha lora_dropout
  ```

* **Prompt Tuning**: Run the prompt tuning script using the preprocessed data to train virtual tokens on the model. 
    ```bash
    python prompt_tuning.py data_path output_path num_vir_tokens num_epochs lr batch_size
    ```

### 3. Generating Content
Run the generation script to create social media posts from summaries:
```bash
python post_generation.py model_path data_path output_path num_posts temp top_p max_new_tokens
```

Use --analyze if you wish to evaluate generated posts using the ROUGE metric.

### 4. Customization and Iteration
The scripts are designed to be easy to iterate with.
* You can run any script with `-h` to see additional parameters for customizing the models and generation process.
* Adjust the number of posts, temperature, top-p sampling, and maximum tokens using the command-line arguments to explore different outputs.
