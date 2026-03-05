import os
import re
import nltk
import torch
import argparse
import pandas as pd
from datasets import Dataset
from playwright.sync_api import sync_playwright
from transformers import pipeline


# Uncomment the line below if running for the first time to download punkt tokenizer
### nltk.download('punkt_tab')

# Set device
device = 0 if torch.cuda.is_available() else -1

# Reads in data from excel file
def load_data(file_path):
    return pd.read_excel(file_path)

# Scrapes article text from given URL
def scrape_article(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            page.goto(url, timeout=15000)
            page.wait_for_timeout(5000)

            # Single article required a button click to reveal article content
            if url == "https://www.capitalgroup.com/advisor/insights/articles/ir-plan-review-tips.html":
                try:
                    button = page.query_selector("text='Tackle the big 3'")
                    if button:
                        button.click()
                        page.wait_for_timeout(5000)
                except:
                    pass

            # Check URL for valid article
            if "/articles/" not in page.url:
                browser.close()
                return None

            # Sections of the article to query
            containers = page.query_selector_all(
                'div[class*="cmp-contentfragment--insights__articlefragment"], div.cmp-text, article'
            )

            # Scrape paragraphs from main containers
            text_blocks = []
            for container in containers:
                paragraphs = container.query_selector_all("p")
                for p_tag in paragraphs:
                    text = p_tag.inner_text().strip()
                    if not text:
                        continue
                    # Stop reading current paragraph if disclosures found
                    if "Read important disclosures" in text:
                        break
                    text_blocks.append(text)

                # Stop reading completely if disclosures found
                if any("Read important disclosures" in t for t in text_blocks):
                    break
            
            # Join and return scraped text
            full_text = "\n\n".join(text_blocks)
            browser.close()
            return full_text
        
        except:
            browser.close()
            return None

# Cleans scraped text
def clean_text(text):
    # Replace non-breaking spaces with normal spaces
    text = text.replace('\u00A0', ' ').replace("&nbsp;", " ").replace("\xa0", " ")

    # Remove unwanted sections such as footnotes, disclaimers, etc.
    end_markers = [
        "Read important disclosures",
        "Bloomberg® is a trademark",
        "The market indexes are unmanaged",
        "Copyright ©",
        "All rights reserved",
        "S&P 500 Index is a market",
        "Investing outside the United States involves risks",
        "Don't miss our latest insights",
        "Hear more on this topic",
        "While money market funds seek to maintain"
    ]
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
            break
    
    # Remove leading/trailing whitespace for each paragraph
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

    # Join paragraphs all in one line
    text = " ".join(paragraphs)

    # Collapse multiple spaces/tabs into a single space
    text = re.sub(r'[ \t]+', ' ', text).strip()

    return text

# Separates text into chunks for processing
def chunk_text(text, max_tokens=400):
    words = text.split()
    result = []
    for i in range(0, len(words), max_tokens):
        result.append(' '.join(words[i:i + max_tokens]))
    return result 



# Instantiate summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

# Summarizes text
def summarize(batch):
    summaries = []
    for text in batch['scraped_text']:
        chunks = chunk_text(text)
        chunk_summaries = summarizer(chunks, min_length=50, max_length=150, do_sample=False)
        combined_summary = ' '.join([s['summary_text'] for s in chunk_summaries])
        summaries.append(combined_summary)
    return {'summary': summaries}

def main(path):
    # Load data
    train_data = load_data(path)

    # Group by unique URLs
    unique_urls = train_data['URL'].dropna().unique()

    # Scrape articles
    url_to_text = {}
    for url in unique_urls:
        try:
            url_to_text[url] = scrape_article(url)
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            url_to_text[url] = None
    
    # Map scraped texts back to the DataFrame
    train_data['scraped_text'] = train_data['URL'].map(url_to_text)

    # Separate entries with missing scraped text
    complete_data = train_data[train_data['scraped_text'].notnull()].copy()
    
    # Preprocess scraped content
    complete_data['scraped_text'] = complete_data['scraped_text'].apply(
        lambda x: clean_text(x) if isinstance(x, str) else x
    )
    
    # Use HuggingFace Dataset for efficient processing
    dataset = Dataset.from_pandas(complete_data[['scraped_text']])

    # Map summarization over the dataset with batching
    dataset = dataset.map(summarize, batched=True, batch_size=8)

    # Add summaries back to DataFrame
    complete_data['summary'] = dataset['summary']

    # Save to new Excel file
    complete_data.to_excel(f'processed_{os.path.basename(path)}.xlsx', index=False)


import sys
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script that will scrape content given data that contains URLs')
    parser.add_argument('data_path', type=str, help='Set data path')
    
    args = parser.parse_args()
    
    main(args.data_path)