import json
import os
import re
import argparse
import sys

def clean_text_for_fasttext(text, version):
    if version == "V1":
        # version used for experiments
        text = text.replace('\n', '').strip().lower()
    elif version == "V2":
        text = re.sub(r'\s+', ' ', text)
        text = text.strip().lower()
    elif version == "V3":
        text = re.sub(r'\r?\n', f' {"<NEW_LINE>"} ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip().lower()
    else:
        raise ValueError(f"Version {version} not supported.")
    return text

def convert_to_fasttext_format(entry, version):
    label = entry['label']
    text = clean_text_for_fasttext(entry['text'], version)
    return f"__label__{label} {text}"

def load_jsonl_dataset(jsonl_path):
    data = []
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as file:
            for line_number, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    data.append(entry)
                except json.JSONDecodeError as e:
                    print(f"JSON decode error in {jsonl_path} at line {line_number}: {e}")
    except FileNotFoundError:
        print(f"Error: File not found - {jsonl_path}")
    except Exception as e:
        print(f"Unexpected error while loading {jsonl_path}: {e}")
    return data

def save_fasttext_split(data, file_path, version):
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            for entry in data:
                fasttext_sample = convert_to_fasttext_format(entry, version)
                if fasttext_sample:
                    file.write(f"{fasttext_sample}\n")
        print(f"Saved {len(data)} samples to {file_path}")
    except Exception as e:
        print(f"Failed to save FastText split to {file_path}: {e}")

def prepare_fasttext_dataset(json_input_path, output_base_path, version):
    datasets = ["all_training_100.jsonl", "train_80.jsonl", "valid_10.jsonl", "test_10.jsonl"]
    for dataset in datasets:
        input_file = os.path.join(json_input_path, dataset)
        if not os.path.exists(input_file):
            print(f"Warning: Dataset file {input_file} does not exist. Skipping.")
            continue

        data = load_jsonl_dataset(input_file)
        if not data:
            print(f"No data loaded from {input_file}. Skipping.")
            continue

        print(f"Loaded {len(data)} samples from {input_file}")
        output_file = os.path.join(output_base_path, dataset.replace(".jsonl", ".txt"))
        save_fasttext_split(data, output_file, version)

def main():
    parser = argparse.ArgumentParser(description="Prepare FastText datasets from JSONL files.")
    parser.add_argument(
        '--version',
        type=str,
        choices=['V1', 'V2', 'V3'],
        default='V1',
        help='Version of text cleaning to apply (default: V1).'
    )
    parser.add_argument(
        '--json_base_path',
        type=str,
        help='data base path '
    )
    parser.add_argument(
        '--output_base_path',
        type=str,
        help='output base path '
    )
    args = parser.parse_args()

    version = args.version
    dataset_types = [ "mkcp", "mkc" ]
    for dataset_type in dataset_types:
        json_base_path = os.path.join(args.json_base_path, dataset_type)
        output_base_path = os.path.join(args.output_base_path, dataset_type, version)

        if not os.path.exists(json_base_path):
            print(f"Warning: JSON base path {json_base_path} does not exist. Skipping dataset type '{dataset_type}'.")
            continue

        for lang_dir in os.listdir(json_base_path):
            lang_dir_path = os.path.join(json_base_path, lang_dir)
            lang_output_base_path = os.path.join(output_base_path, lang_dir)
            if os.path.exists(lang_output_base_path):
                print(f"Train data already exists for {lang_output_base_path}")
                continue

            os.makedirs(lang_output_base_path, exist_ok=True)
            if not os.path.isdir(lang_dir_path):
                print(f"Skipping non-directory entry: {lang_dir_path}")
                continue

            print(f"Preparing FastText dataset for language: {lang_dir}")
            prepare_fasttext_dataset(
                json_input_path=lang_dir_path,
                output_base_path=lang_output_base_path,
                version=version
            )
            print(f"Successfully prepared FastText dataset for {lang_dir}.\n")

if __name__ == "__main__":
    main()