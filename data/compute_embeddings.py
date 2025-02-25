from datatrove.pipeline.readers import ParquetReader, JsonlReader
from datatrove.pipeline.writers import ParquetWriter
from datatrove.executor.local import LocalPipelineExecutor
from fineweb2_hq.utils import list_files
from fineweb2_hq.xlmr_embeddings import XLMRobertaEmbeddingAnnotator
import os
import argparse


def main(args):
    if args.reader_type == "parquet":
        reader = ParquetReader(
            args.input_dir,
        )
    elif args.reader_type == "jsonl":
        reader = JsonlReader(
            args.input_dir,
        )

    pipeline = [
        reader,
        XLMRobertaEmbeddingAnnotator(
            model="FacebookAI/xlm-roberta-base",
            tokenizer_batch_size=10000,
            model_batch_size=4096,
        ),
        ParquetWriter(
            args.output_dir,
            compression="zstd",
        ),
    ]

    LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=len(list_files(args.input_dir)) if os.path.isdir(args.input_dir) else 1,
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
        "--reader-type",
        help="type of the reader",
        choices=["jsonl", "parquet"],
        default="parquet",
    )
    parser.add_argument("--num-workers", help="number of workers", default=2, type=int)

    args = parser.parse_args()

    main(args)
