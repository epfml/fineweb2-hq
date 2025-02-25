from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import ParquetWriter
from datatrove.executor.local import LocalPipelineExecutor
from fineweb2_hq.mlp import (
    estimate_classifier_threshold,
    EmbeddingBinaryClassifierFilter,
    BinaryClassifier,
)
from fineweb2_hq.utils import list_files
import argparse


def main(args):
    classifier = BinaryClassifier.from_pt(args.classifier_path)

    threshold = estimate_classifier_threshold(
        args.input_dir,
        classifier,
        retention_rate=args.retention_rate,
        num_samples=1_000_000,
    )
    print(f"Estimated threshold: {threshold:0.8f}")

    pipeline = [
        ParquetReader(
            args.input_dir,
        ),
        EmbeddingBinaryClassifierFilter(
            classifier=classifier,
            threshold=threshold,
            batch_size=10_000,
        ),
        ParquetWriter(
            args.output_dir,
            compression="zstd",
        ),
    ]

    LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=len(list_files(args.input_dir)),
        workers=args.num_workers,
        logging_dir=args.logs_dir,
    ).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir", help="directory of the input files", required=True
    )
    parser.add_argument(
        "--output-dir", help="directory of the processed output files", required=True
    )
    parser.add_argument(
        "--logs-dir",
        help="directory of the logs",
        default=None,
    )
    parser.add_argument(
        "--classifier-path", help="path of the MLP classifier", required=True
    )
    parser.add_argument(
        "--retention-rate", help="retention rate", default=0.1, type=float
    )
    parser.add_argument("--num-workers", help="number of workers", default=2, type=int)

    args = parser.parse_args()

    main(args)
