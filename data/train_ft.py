import os
import pandas as pd
import random
import math
import csv
import argparse
import time
from collections import defaultdict

random.seed(42)

def load_txt_dataset(txt_file_path):
    data = []
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line.startswith("__label__"):
                try:
                    label, text = line.split(' ', 1)
                    text = text.strip()
                    if label not in ["__label__positive", "__label__negative"]:
                        print(f"Warning: Unknown label '{label}' at line {line_num}. Skipping.")
                        continue
                    data.append({
                        'text': text,
                        'label': label
                    })
                except ValueError:
                    print(f"Warning: Line {line_num} is not properly formatted. Skipping.")
            else:
                print(f"Warning: Line {line_num} does not start with a label. Skipping.")
    return pd.DataFrame(data)

def train_fasttext_model(
        training_file,
        validation_file,
        output_path,
        lang,
        loss='softmax',
        autotune_metric='f1',
        autotune_duration=300):
    import fasttext
    start_time = time.time()
    N_GRAMS = defaultdict(lambda: 2, {"cmn_Hani": 4})
    if lang == "cmn_Hani":
        model = fasttext.train_supervised(
        input=training_file,
        epoch=30,
        lr=0.1,
        wordNgrams=N_GRAMS[lang],
        minCount=0,
        loss=loss,
        )
    else:
        model = fasttext.train_supervised(
                input=training_file,
                wordNgrams=N_GRAMS[lang],
                minCount=1,
                loss=loss,
                autotuneValidationFile=validation_file,
                autotuneMetric=autotune_metric,
                autotuneDuration=autotune_duration
            )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Autotuning completed in {elapsed_time:.2f} seconds.")

    model_filename = f'autotuned_fasttext_model_{lang}.bin'
    full_path = os.path.join(output_path, model_filename)
    if os.path.exists(full_path):
        raise FileExistsError(f"The path '{full_path}' already exists.")
    model.save_model(full_path)
    print(f"Autotuned fastText model saved as '{model_filename}'.")

    return model

def compute_fasttext_scores(fasttext_model, texts, positive_label="__label__positive", batch_size=1000):
    scores = []
    print("Computing fastText prediction scores...")
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        predictions = fasttext_model.predict(batch_texts, k=-1)
        for labels, probs in zip(*predictions):
            if positive_label in labels:
                pos_index = labels.index(positive_label)
                score = probs[pos_index]
            else:
                score = 0.0
            scores.append(score)
        print(f"Processed batch {i // batch_size + 1} / {math.ceil(len(texts) / batch_size)}")
    return scores

def save_selected_samples(output_path, data_plot, language_code):
    top_high_scores = data_plot.nlargest(25, 'score')
    top_low_scores = data_plot.nsmallest(25, 'score')
    
    output_file = f'selected_samples_{language_code}.txt'
    with open(os.path.join(output_path, output_file), 'w', encoding='utf-8') as f:
        
        f.write("\n=== Top 25 Items with Highest Scores ===\n\n")
        for idx, row in top_high_scores.iterrows():
            f.write(f"Index: {idx}\n")
            f.write(f"Score: {row['score']}\n")
            f.write(f"Text: {row['text']}\n")
            f.write("-" * 80 + "\n")
        
        f.write("\n=== Top 25 Items with Lowest Scores ===\n\n")
        for idx, row in top_low_scores.iterrows():
            f.write(f"Index: {idx}\n")
            f.write(f"Score: {row['score']}\n")
            f.write(f"Text: {row['text']}\n")
            f.write("-" * 80 + "\n")
    print(f"Selected samples have been saved to '{output_file}'.")

def process_language(base_path, lang):
    lang_output_path = os.path.join(base_path, lang)
    training_file = os.path.join(base_path, lang, 'train_80.txt')
    validation_file = os.path.join(base_path, lang, 'valid_10.txt')
    test_file = os.path.join(base_path, lang, 'test_10.txt')
    os.makedirs(lang_output_path, exist_ok=True)

    print(f"\nProcessing language: {lang}")
    print("Training fastText supervised model...")
    fasttext_model = train_fasttext_model(
        training_file=training_file,
        validation_file=validation_file,
        output_path=lang_output_path,
        lang=lang,
        wordNgrams=2
    )

    print("Loading labeled TXT dataset...")
    if not os.path.exists(test_file):
        raise ValueError(f"Error: Test file '{test_file}' does not exist.")
    data = load_txt_dataset(test_file)
    print(f"Total labeled texts found: {len(data)}")
    fineweb_data = data[data['label'] == '__label__negative'].copy()

    print("Computing fastText scores...")
    scores = compute_fasttext_scores(fasttext_model, fineweb_data['text'].tolist())
    fineweb_data['score'] = scores
    print("fastText scores computation completed.")
    save_selected_samples(lang_output_path, fineweb_data, lang)

def parse_args():
    parser = argparse.ArgumentParser(description="fastText Training and Sample Selection Script")
    parser.add_argument(
        '--type', 
        type=str, 
        default='mkcp', 
        help='Type identifier (default: mkcp)'
    )
    parser.add_argument(
        '--csv_file',
        type=str,
        help='Path to the language mappings CSV file'
    )
    parser.add_argument(
        '--base_input_path',
        type=str,
        help='input path'
    )
    return parser.parse_args()

def main():
    args = parse_args()

    csv_file_path = args.csv_file
    base_path = os.path.join(args.base_input_path)
    languages = []
            
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file '{csv_file_path}' does not exist.")
        return

    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row_num, row in enumerate(reader, 1):
            try:
                lang_code_mFineweb = row['fineweb2'].strip()
                lang_iso1 = row['iso1'].strip()
                languages.append((lang_code_mFineweb, lang_iso1))
            except KeyError as e:
                print(f"Warning: Missing column {e} in CSV at row {row_num}. Skipping this row.")
            
    if not languages:
        print(f"No languages found in the CSV file '{csv_file_path}'. Exiting.")
        return
    print(f"Found {len(languages)} languages in the CSV.")
    for lang, iso1 in languages:
        try:
            process_language(base_path, lang)
        except Exception as e:
            print(f"An error occurred while processing language '{lang}' for {type} and '{version}': {e}")

if __name__ == "__main__":
    main()