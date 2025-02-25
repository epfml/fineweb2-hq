# Training

We provide the config used for training the LLM models using the [`nanotron`](https://github.com/huggingface/nanotron) library.

To tokenize the data for training, run the following command:
```bash
python tokenize_data.py --input-dir ../data/fineweb2-hq/fra_Latn --output-dir ./fineweb2-hq-tokenized/fra_Latn --rehydrate
```