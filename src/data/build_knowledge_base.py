"""
Script to extract and curate historical resolution pairs for @AppleSupport.
Reads from the HuggingFace mirror of the Customer Support on Twitter dataset,
filters for AppleSupport, extracts (inquiry, resolution) pairs, and saves a curated
knowledge base for retrieval-augmented generation.
"""

import re
import json
import sys
from pathlib import Path
import polars as pl

sys.stdout.reconfigure(encoding='utf-8')

DATA_URL = "https://huggingface.co/datasets/TNE-AI/customer-support-on-twitter-conversation/resolve/main/data/train-00000-of-00001.parquet"
OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "historical_resolutions.json"

def clean_tweet_text(text: str) -> str:
    """Clean and normalize raw tweet text."""
    if not text:
        return ""
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def parse_conversation(conversation_text: str):
    lines = conversation_text.strip().split("\n")
    customer_msgs = []
    support_msgs = []

    current_speaker = None
    current_buffer = []

    for line in lines:
        line_clean = line.strip()
        if line_clean.startswith("Customer:"):
            if current_speaker == "Support" and current_buffer:
                support_msgs.append(" ".join(current_buffer))
                current_buffer = []
            current_speaker = "Customer"
            current_buffer.append(line_clean[len("Customer:"):].strip())
        elif line_clean.startswith("Support:"):
            if current_speaker == "Customer" and current_buffer:
                customer_msgs.append(" ".join(current_buffer))
                current_buffer = []
            current_speaker = "Support"
            current_buffer.append(line_clean[len("Support:"):].strip())
        else:
            if current_buffer:
                current_buffer.append(line_clean)

    if current_speaker == "Support" and current_buffer:
        support_msgs.append(" ".join(current_buffer))
    elif current_speaker == "Customer" and current_buffer:
        customer_msgs.append(" ".join(current_buffer))

    if customer_msgs and support_msgs:
        inquiry = clean_tweet_text(customer_msgs[0])
        reply = clean_tweet_text(support_msgs[0])
        return inquiry, reply
    return None, None

def main():
    print(f"Loading conversations from {DATA_URL}...")
    df = pl.read_parquet(DATA_URL)
    apple_df = df.filter(pl.col("company") == "AppleSupport")
    print(f"Total AppleSupport threads found: {len(apple_df)}")

    curated = []
    seen_inquiries = set()

    for row in apple_df.iter_rows(named=True):
        conv = row.get("conversation", "")
        inquiry, reply = parse_conversation(conv)
        if not inquiry or not reply:
            continue
        if len(inquiry) < 15 or len(reply) < 15:
            continue
        if inquiry in seen_inquiries:
            continue

        seen_inquiries.add(inquiry)
        curated.append({
            "conversation_id": row.get("conversation_id"),
            "customer_inquiry": inquiry,
            "support_resolution": reply
        })

        if len(curated) >= 1500:
            break

    print(f"Extracted {len(curated)} high-quality historical resolution pairs.")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(curated, f, indent=2, ensure_ascii=False)
    print(f"Saved historical resolutions knowledge base to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
