from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import ParquetWriter
from datatrove.executor.local import LocalPipelineExecutor
import argparse
from fineweb2_hq.utils import list_files
from fineweb2_hq.cs import (
    EmbeddingCosineSimilarityFilter,
    estimate_cosine_threshold,
    load_training_embeddings,
)


def main(args):
    training_data_embeddings = load_training_embeddings(
        f"{args.dataset_dir}/train_80/",
        num_samples=8192,
        seed=42,
    )

    threshold = estimate_cosine_threshold(
        args.input_dir,
        training_data_embeddings,
        retention_rate=args.retention_rate,
        num_samples=1_000_000,
    )
    print(f"Estimated threshold: {threshold:0.8f}")

    pipeline = [
        ParquetReader(
            args.input_dir,
        ),
        EmbeddingCosineSimilarityFilter(
            threshold=threshold,
            batch_size=10_000,
            positive_embeddings=training_data_embeddings,
            embedding_key=lambda x: x.metadata["embeddings"][0],
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
        "--dataset-dir",
        help="directory of the MKC/MKC+ dataset with embeddings",
        required=True,
    )
    parser.add_argument(
        "--retention-rate", help="retention rate", default=0.1, type=float
    )
    parser.add_argument("--num-workers", help="number of workers", default=2, type=int)

    args = parser.parse_args()

    main(args)
