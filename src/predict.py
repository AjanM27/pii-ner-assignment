import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from labels import ID2LABEL, label_is_pii
import os
import re


# ---------------------------
#   VALIDATION RULES
# ---------------------------

def validate_email(text):
    """Email must contain: something at something dot something"""
    t = text.lower()
    return (" at " in t) and (" dot " in t)


def validate_phone(text):
    """Must be at least 10 digits OR 10 spoken numbers."""
    digits = re.findall(r'[0-9]', text)
    if len(digits) >= 10:
        return True

    spoken = text.split()
    digit_words = {
        "zero","one","two","three","four","five","six","seven","eight","nine"
    }
    spoken_count = sum(w in digit_words for w in spoken)
    return spoken_count >= 10


def validate_credit_card(text):
    """Must be 16 digits (spoken or numeric)."""
    digits = re.findall(r'[0-9]', text)
    if len(digits) == 16:
        return True

    spoken = text.split()
    digit_words = {
        "zero","one","two","three","four","five","six","seven","eight","nine"
    }
    spoken_count = sum(w in digit_words for w in spoken)
    return spoken_count == 16


def validate_date(text):
    """Rough check: must have 2–4 numeric tokens."""
    parts = re.findall(r'[0-9]+', text)
    return 2 <= len(parts) <= 4


# ---------------------------
#   POST-PROCESSING
# ---------------------------

def clean_and_filter_spans(text, spans):
    """
    Takes raw spans (start, end, label) and removes invalid PII.
    """
    cleaned = []
    for s, e, lab in spans:
        frag = text[s:e]

        # Only apply validation to PII labels
        if lab == "EMAIL" and not validate_email(frag):
            continue
        if lab == "PHONE" and not validate_phone(frag):
            continue
        if lab == "CREDIT_CARD" and not validate_credit_card(frag):
            continue
        if lab == "DATE" and not validate_date(frag):
            continue

        cleaned.append((s, e, lab))

    return merge_overlapping_spans(cleaned)


def merge_overlapping_spans(spans):
    """Merges overlapping spans with the same label."""
    if not spans:
        return spans

    spans = sorted(spans)
    merged = []
    cur_s, cur_e, cur_l = spans[0]

    for s, e, lab in spans[1:]:
        if lab == cur_l and s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e, cur_l))
            cur_s, cur_e, cur_l = s, e, lab

    merged.append((cur_s, cur_e, cur_l))
    return merged


# ---------------------------
#   BIO → SPANS
# ---------------------------
def bio_to_spans(text, offsets, label_ids):
    """
    Improved BIO decoding:
    - merges fragmented spans
    - fixes boundary splits from wordpiece tokenization
    - handles adjacent B/I inconsistencies
    - guarantees sorted, non-overlapping spans
    """

    spans = []
    cur_label = None
    cur_start = None
    cur_end = None

    for (start, end), lid in zip(offsets, label_ids):

        # Skip special tokens like CLS/SEP
        if start == 0 and end == 0:
            continue

        tag = ID2LABEL.get(int(lid), "O")

        if tag == "O":
            # If we were inside a span, close it
            if cur_label is not None:
                spans.append((cur_start, cur_end, cur_label))
                cur_label = None
            continue

        prefix, ent_type = tag.split("-", 1)

        # Case 1: Begin new entity
        if prefix == "B":
            # Close previous
            if cur_label is not None:
                spans.append((cur_start, cur_end, cur_label))

            cur_label = ent_type
            cur_start = start
            cur_end = end
            continue

        # Case 2: Inside entity
        if prefix == "I":
            if cur_label == ent_type:
                # Continuation of same span
                cur_end = end
            else:
                # I-X after a B-Y → treat as new span
                if cur_label is not None:
                    spans.append((cur_start, cur_end, cur_label))

                cur_label = ent_type
                cur_start = start
                cur_end = end
            continue

    # Close final span
    if cur_label is not None:
        spans.append((cur_start, cur_end, cur_label))

    # ---- EXTRA FIXES BELOW ----

    # Fix tokenizer-driven fragmenting (merge adjacent spans w/ same label)
    merged = []
    for s, e, lab in sorted(spans):
        if merged and merged[-1][2] == lab and s <= merged[-1][1] + 1:
            # merge overlapping + adjacent
            prev_s, prev_e, prev_l = merged[-1]
            merged[-1] = (prev_s, max(prev_e, e), prev_l)
        else:
            merged.append((s, e, lab))

    return merged



# ---------------------------
#   MAIN INFERENCE PIPELINE
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir if args.model_name is None else args.model_name
    )
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(args.device)
    model.eval()

    results = {}

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            uid = obj["id"]

            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            offsets = enc["offset_mapping"][0].tolist()
            input_ids = enc["input_ids"].to(args.device)
            attention_mask = enc["attention_mask"].to(args.device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits[0]
                pred_ids = logits.argmax(dim=-1).cpu().tolist()

            raw_spans = bio_to_spans(text, offsets, pred_ids)

            # 🔥 APPLY POST-PROCESSING FIXES HERE
            final_spans = clean_and_filter_spans(text, raw_spans)

            ents = []
            for s, e, lab in final_spans:
                ents.append(
                    {
                        "start": int(s),
                        "end": int(e),
                        "label": lab,
                        "pii": bool(label_is_pii(lab)),
                    }
                )

            results[uid] = ents

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote predictions for {len(results)} utterances to {args.output}")


if __name__ == "__main__":
    main()
