import csv
import random
import os
import json
from sklearn.model_selection import train_test_split
from glob import glob
from datasets import load_dataset
from tqdm import tqdm
import re
from collections import defaultdict
import pandas as pd
import argparse

import ftfy

# FROM https://github.com/huggingface/datatrove/commit/0c891f6dbbc0297b294fa47e772998df21bdb7c4#diff-97a210017d6b2669f47e2a89312c1a469efc44e77584d616b64c612be52e8cd8R61
unescape_html = "auto"
remove_terminal_escapes = True
fix_encoding = True
restore_byte_a0 = True
replace_lossy_sequences = True
decode_inconsistent_utf8 = True
fix_c1_controls = True
fix_latin_ligatures = False
fix_character_width = False
uncurl_quotes = False
fix_line_breaks = False
fix_surrogates = True
remove_control_chars = True
normalization = None
FTFT_CONFIG = ftfy.TextFixerConfig(
    unescape_html=unescape_html,
    remove_terminal_escapes=remove_terminal_escapes,
    fix_encoding=fix_encoding,
    restore_byte_a0=restore_byte_a0,
    replace_lossy_sequences=replace_lossy_sequences,
    decode_inconsistent_utf8=decode_inconsistent_utf8,
    fix_c1_controls=fix_c1_controls,
    fix_latin_ligatures=fix_latin_ligatures,
    fix_character_width=fix_character_width,
    uncurl_quotes=uncurl_quotes,
    fix_line_breaks=fix_line_breaks,
    fix_surrogates=fix_surrogates,
    remove_control_chars=remove_control_chars,
    normalization=normalization,
)


def create_json_entry(sample, label):
    return {"label": label, "text": sample}


def normalize_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = ftfy.fix_text(text, config=FTFT_CONFIG)
    return text


def load_aya_instruct_data(
    dataset_name, split, sample_size, data_files=None, subname=None
):
    dataset = load_dataset(dataset_name, subname, split=split, data_files=data_files)
    df = dataset.to_pandas()

    text_columns = ["inputs", "targets"]
    for col in text_columns:
        df[col] = df[col].apply(normalize_text)

    samples = to_aya_text([item for item in dataset if item])
    samples = [s for s in samples if "<unk>" not in s]
    if sample_size and len(samples) > sample_size:
        samples = random.sample(list(samples), sample_size)
    return samples


def load_aya_human(dataset_name, langauge, split):
    dataset = load_dataset(dataset_name, split=split).shuffle(seed=42)
    samples = to_aya_text(
        [item for item in dataset if item["language_code"] == langauge]
    )
    return samples


def load_fineweb2_samples(dataset_path, sample_size, file_sample_size=3):
    pattern = os.path.join(dataset_path, "*.parquet")
    data_files = glob(pattern)

    if file_sample_size is not None:
        total_files = len(data_files)
        if file_sample_size > total_files:
            raise ValueError(
                f"file_sample_size ({file_sample_size}) exceeds the total number of files ({total_files})."
            )
        data_files = random.sample(data_files, file_sample_size)
        print(
            f"Randomly selected {file_sample_size} out of {total_files} Parquet files to process."
        )
    else:
        print(f"Processing all {len(data_files)} Parquet files.")

    samples = []
    for file_path in data_files:
        print(f"Processing file: {file_path}")
        df = pd.read_parquet(file_path, columns=["text"])
        df_na = df.dropna(subset=["text"])
        if len(df) != len(df_na):
            breakpoint()

        print(f"Loaded {len(df)} samples")
        for text in tqdm(
            df["text"], desc=f"Reading {os.path.basename(file_path)}", unit=" lines"
        ):
            samples.append(text)

    print(f"\nLoaded {len(samples)} 'text' entries from the dataset.")
    if len(samples) > sample_size:
        samples = random.sample(samples, sample_size)
        print(f"Randomly sampled {sample_size} 'text' entries.")
    return samples


def to_aya_text(dataset):
    aya_samples = [
        f"{sample['inputs']} {sample['targets'][:-3]}"
        if sample["targets"].endswith("...")
        else f"{sample['inputs']} {sample['targets']}"
        for sample in dataset
    ]
    return aya_samples


def load_json_samples(json_path, target_language, sample_size=None):
    with open(json_path, "r") as file:
        data = json.load(file)

    samples = [
        f"{item['question']} {item['options'][item['answer'] - 1]}"
        for item in data
        if item["language"] == target_language
    ]

    if sample_size and len(samples) > sample_size:
        samples = random.sample(samples, sample_size)

    return samples


def load_include_from_huggingface(dataset_name, target_language, sample_size=None):
    dataset = load_dataset(dataset_name, name=target_language)
    data_split = dataset["test"]
    samples = [
        f"{item['question']} {item['choices'][item['answer'] - 1]}"
        for item in data_split
    ]
    if sample_size and len(samples) > sample_size:
        samples = random.sample(samples, sample_size)

    return samples


def to_wikihow_text(dataset):
    return [sample["text"] for sample in dataset]


def load_oasst2_samples(lang_code, sample_size=None, max_conversations_per_root=2):
    print("Loading dataset...")
    dataset = load_dataset("OpenAssistant/oasst2", split="train")
    filtered_dataset = dataset.filter(lambda x: x["lang"] == lang_code)
    print(
        f"Filtered dataset to {len(filtered_dataset)} samples for language '{lang_code}'."
    )

    print("Building message dictionary...")
    message_dict = {sample["message_id"]: sample for sample in filtered_dataset}
    print(f"Created message_dict with {len(message_dict)} entries.")

    print("Identifying last messages...")
    parent_ids = set(
        sample["parent_id"] for sample in filtered_dataset if sample["parent_id"]
    )
    last_messages = [
        sample for sample in filtered_dataset if sample["message_id"] not in parent_ids
    ]
    print(f"Found {len(last_messages)} last messages.")

    print("Traversing conversations to build complete conversation texts...")
    conversations = []
    for idx, sample in enumerate(last_messages, 1):
        texts = []
        current_sample = sample
        visited = set()

        while current_sample:
            msg_id = current_sample["message_id"]
            if msg_id in visited:
                print(
                    f"Cycle detected at message_id: {msg_id}. Skipping this conversation."
                )
                break
            visited.add(msg_id)

            texts.insert(0, current_sample["text"])

            parent_id = current_sample["parent_id"]

            if parent_id and parent_id in message_dict:
                current_sample = message_dict[parent_id]
            else:
                root_id = current_sample["message_id"]
                current_sample = None

        if current_sample is None and (not parent_id or parent_id not in message_dict):
            concatenated_text = " ".join(texts)
            conversations.append((root_id, concatenated_text))

        if idx % 10000 == 0:
            print(f"Processed {idx} / {len(last_messages)} conversations...")

    print(f"Total conversations collected: {len(conversations)}")
    print("Grouping conversations by root_id...")
    root_to_conversations = defaultdict(list)
    for root_id, convo_text in conversations:
        root_to_conversations[root_id].append(convo_text)

    print(f"Total unique roots: {len(root_to_conversations)}")
    print(f"Selecting up to {max_conversations_per_root} conversations per root...")
    selected_conversations = []
    for root_id, convo_list in root_to_conversations.items():
        if len(convo_list) <= max_conversations_per_root:
            selected_conversations.extend(convo_list)
        else:
            selected_conversations.extend(
                random.sample(convo_list, max_conversations_per_root)
            )

    print(
        f"Total selected conversations after limiting per root: {len(selected_conversations)}"
    )
    if sample_size and len(selected_conversations) > sample_size:
        print(f"Sampling {sample_size} conversations from the selected pool...")
        selected_conversations = random.sample(selected_conversations, sample_size)
        print(f"Final number of conversations: {len(selected_conversations)}")
    else:
        print("No additional sampling needed based on sample_size.")

    return selected_conversations


def load_openai_mmlu_samples(lang_code, sample_size):
    dataset = load_dataset("openai/MMMLU", lang_code)
    concatenated_samples = []

    for sample in dataset["test"]:
        question = sample.get("Question", "").strip()
        answer_key = sample.get("Answer", "").strip()
        answer_text = sample.get(answer_key, "").strip()

        concatenated_text = f"{question} {answer_text}"
        concatenated_samples.append(concatenated_text)

    if sample_size and len(concatenated_samples) > sample_size:
        concatenated_samples = random.sample(concatenated_samples, sample_size)

    return concatenated_samples


def generate_json_dataset(
    lang_code_epfl_language,
    lang_code_aya,
    lang_code_aya_human,
    lang_code_openai,
    lang_code_openass,
    lang_code_fineweb2,
    num_samples,
    fineweb2_path,
    output_base_path,
):
    all_samples = []

    # Load OpenAI MMLU Samples
    if lang_code_openai:
        openai_mmlu_samples = load_openai_mmlu_samples(
            lang_code_openai,
            sample_size=num_samples,
        )
        print(
            f"Loaded {len(openai_mmlu_samples)} OpenAI MMLU samples for {lang_code_openai}."
        )
        all_samples += [create_json_entry(s, "positive") for s in openai_mmlu_samples]

    # Load Lite Samples
    if lang_code_epfl_language:
        lite_samples = load_include_from_huggingface(
            dataset_name="CohereForAI/include-base-44",
            target_language=lang_code_epfl_language,
            sample_size=num_samples,
        )
        print(f"Loaded {len(lite_samples)} Lite samples for {lang_code_epfl_language}.")
        all_samples += [create_json_entry(s, "positive") for s in lite_samples]

    # Load OpenAssistant2 Samples
    if lang_code_openass:
        oasst2_samples = load_oasst2_samples(lang_code_openass, sample_size=num_samples)
        print(
            f"Loaded {len(oasst2_samples)} OpenAssistant2 samples for {lang_code_openass}."
        )
        all_samples += [create_json_entry(s, "positive") for s in oasst2_samples]

    # Load AYA Samples
    if lang_code_aya_human:
        aya_human_samples = load_aya_human(
            dataset_name="cohereForAI/aya_dataset",
            langauge=lang_code_aya_human,
            split="train",
        )
        all_samples += [create_json_entry(s, "positive") for s in aya_human_samples]
        print(
            f"Loaded {len(aya_human_samples)} AYA human samples for {lang_code_aya_human}."
        )

    if lang_code_aya:
        aya_samples = load_aya_instruct_data(
            dataset_name="cohereForAI/aya_collection_language_split",
            subname=lang_code_aya,
            split="train",
            sample_size=num_samples - len(all_samples),
        )
        all_samples += [create_json_entry(s, "positive") for s in aya_samples]
        print(f"Loaded {len(aya_samples)} AYA samples for {lang_code_aya}.")

    print(f"Total {len(all_samples)} positive samples for {lang_code_fineweb2}.")

    fineweb_samples = load_fineweb2_samples(
        f"{fineweb2_path}/{lang_code_fineweb2}/train", len(all_samples)
    )
    all_samples += [create_json_entry(s, "negative") for s in fineweb_samples]
    print(f"Added {len(fineweb_samples)} negative samples for {lang_code_fineweb2}.")

    random.shuffle(all_samples)

    os.makedirs(output_base_path, exist_ok=True)
    all_training_file = os.path.join(output_base_path, "all_training_100.jsonl")

    label_pattern = re.compile(r"^\s*__label__(positive|negative)\s*$", re.IGNORECASE)
    all_labels_only = [
        item for item in all_training_file if bool(label_pattern.match(item))
    ]
    if len(all_labels_only) > 0:
        print("All samples in contain only labels. Triggering breakpoint.")
        breakpoint()

    with open(all_training_file, "w", encoding="utf-8") as file:
        for sample in all_samples:
            json_line = json.dumps(sample, ensure_ascii=False)
            file.write(f"{json_line}\n")
    print(f"All samples have been saved to {all_training_file}")

    labels = [sample["label"] for sample in all_samples]

    train_set, temp_set, _, temp_labels = train_test_split(
        all_samples, labels, test_size=0.2, random_state=42, stratify=labels
    )

    valid_set, test_set, _, _ = train_test_split(
        temp_set, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )

    print(
        f"Split into {len(train_set)} train, {len(valid_set)} valid, and {len(test_set)} test samples for {lang_code_fineweb2}."
    )

    split_files = {
        "train": os.path.join(output_base_path, "train_80.jsonl"),
        "valid": os.path.join(output_base_path, "valid_10.jsonl"),
        "test": os.path.join(output_base_path, "test_10.jsonl"),
    }

    for split_name, file_path in split_files.items():
        if split_name == "train":
            split_data = train_set
        elif split_name == "valid":
            split_data = valid_set
        elif split_name == "test":
            split_data = test_set

        with open(file_path, "w", encoding="utf-8") as file:
            for sample in split_data:
                json_line = json.dumps(sample, ensure_ascii=False)
                file.write(f"{json_line}\n")
        print(f"Saved {split_name} set with {len(split_data)} samples to {file_path}.")
    print("Dataset generation and splitting completed successfully.")


def main(args):
    sample_size = 100_000
    languages = []

    # Read the CSV file and populate the languages list
    with open(args.language_mapping, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            lang_code_epfl_language = row["epfl"].strip()
            lang_code_aya = row["aya"].strip()
            lang_code_aya_human = row["aya_human"].strip()
            lang_code_openai = row["openai"].strip()
            lang_code_openass = row["openassistant"].strip()
            lang_code_fineweb2 = row["fineweb2"].strip()

            languages.append(
                (
                    lang_code_epfl_language,
                    lang_code_aya,
                    lang_code_aya_human,
                    lang_code_openai,
                    lang_code_openass,
                    lang_code_fineweb2,
                )
            )

    print(f"Found {len(languages)} languages in the CSV.")

    for (
        lang_code_epfl_language,
        lang_code_aya,
        lang_code_aya_human,
        lang_code_openai,
        lang_code_openass,
        lang_code_fineweb2,
    ) in languages:
        print(f"Processing language: {lang_code_fineweb2}")
        random.seed(42)

        output_base_path = f"{args.output_dir}/{lang_code_fineweb2}"
        if os.path.exists(output_base_path):
            print(f"Train data already exists for {output_base_path}")
            continue
        os.makedirs(os.path.dirname(output_base_path), exist_ok=True)

        generate_json_dataset(
            lang_code_epfl_language,
            lang_code_aya,
            lang_code_aya_human,
            lang_code_openai,
            lang_code_openass,
            lang_code_fineweb2,
            num_samples=sample_size,
            fineweb2_path=args.fineweb2_path,
            output_base_path=output_base_path,
        )
        print(f"Successfully created JSON dataset for {lang_code_fineweb2}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir", help="directory of the output datasets", required=True
    )
    parser.add_argument(
        "--language-mapping",
        help="directory of the .csv file with the language mapping",
        required=True,
    )
    parser.add_argument(
        "--fineweb2-path", help="directory of the FineWeb2 dataset", required=True
    )

    args = parser.parse_args()

    main(args)
