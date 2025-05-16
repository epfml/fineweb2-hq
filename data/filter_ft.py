import json
import numpy as np
import os
import argparse
import gzip

from data_pipeline_pretrain.executor.slurm_nodes import SlurmPipelineNodeExecutor
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.filters import FastTextClassifierFilter
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers.jsonl import JsonlWriter

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--logging-dir",
        type=str,
        required=True,
        help="folder where the logs will be stored",
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="folder where the input data is",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="folder where the output data will be stored",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="model path",
    )

    parser.add_argument(
        "--keep-percentage",
        type=float,
        required=True,
        help="percentage of the data to keep",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=10_000,
        required=False,
        help="number of samples to estimate threshold",
    )

    parser.add_argument(
        "--num-tasks",
        type=int,
        default=576,
        help="number of tasks that will be launched",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=576,
        help="number of workers (=number of nodes) that will be used for the job",
    )
    args = parser.parse_args()
    return args

args = get_args()

def compute_threshold(
    input_data_path: str,
    temp_threshold_scores_path: str,
    model_url: str,
    keep_percent: float,
    sample_size: int = 10_000,
) -> float:
    temp_score_output_path = os.path.join(temp_threshold_scores_path, "output", str(keep_percent))
    pipeline_sample = [
        ParquetReader(
            data_folder=input_data_path,
            text_key="text",
            limit=sample_size
        ),
        FastTextClassifierFilter(
            model_url=model_url,
            keep_labels=[("positive", -9e9)],
            save_labels_in_metadata=True,
            exclusion_writer=JsonlWriter(os.path.join(temp_threshold_scores_path, "removed", str(keep_percent))),
        ),
        JsonlWriter(temp_score_output_path)
    ]

    executor_sample = LocalPipelineExecutor(
        pipeline=pipeline_sample,
        tasks=1,
        logging_dir=os.path.join(temp_threshold_scores_path, "logs", str(keep_percent)),
        randomize_start_duration=60,
    )
    executor_sample.run()

    scores = []
    with gzip.open(os.path.join(temp_score_output_path, "00000.jsonl.gz"), 'rt', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            score = record.get("metadata", {}).get('__label__positive')
            if score is not None:
                try:
                    scores.append(float(score))
                except ValueError:
                    print(f"Invalid score value encountered: {score}")

    threshold = np.percentile(scores, (1 - keep_percent) * 100)
    print(f"Computed percentile threshold = {threshold}")
    try:
        threshold_output_path = os.path.join(temp_threshold_scores_path, "threshold.txt")
        with open(threshold_output_path, 'w', encoding='utf-8') as f_out:
            f_out.write(f"{threshold}\n")
        print(f"Threshold value saved to {threshold_output_path}")
    except IOError as e:
        print(f"Failed to write threshold to file: {e}")
    return threshold

def process_data():
    threshold = compute_threshold(
            input_data_path=args.input_dir,
            temp_threshold_scores_path=os.path.join(args.logging_dir, "tmp_scores"),
            model_url=args.model_path,
            keep_percent=args.keep_percentage,
            sample_size=args.sample_size,
        )
    pipeline = [
            ParquetReader(
                data_folder=args.input_dir,
                text_key="text",
            ),
            FastTextClassifierFilter(
                model_url=args.model_path,
                keep_labels=[("positive", threshold)],
                save_labels_in_metadata=True,
                exclusion_writer=JsonlWriter(os.path.join(args.output_dir, "removed", "fasttext", str(args.keep_percentage))),
            ),
            JsonlWriter(os.path.join(args.output_dir, "output"))
        ]
    LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=args.num_tasks,
        workers=args.num_workers,
        logging_dir=args.logs_dir,
    ).run()

if __name__ == '__main__':
    process_data()