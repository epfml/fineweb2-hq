from datatrove.pipeline.readers import ParquetReader
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.tokens import DocumentTokenizer
import argparse
from fineweb2_hq.utils import list_files
from fineweb2_hq.rehydrater import Rehydrater


def main(args):
    pipeline = [
        ParquetReader(args.input_dir),
        *([Rehydrater()] if args.rehydrate else []),
        DocumentTokenizer(
            output_folder=args.output_dir,
            tokenizer_name_or_path="mistralai/Mistral-Nemo-Base-2407",
            max_tokens_per_file=1e9,
            shuffle=False,
            eos_token=None,
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
        "--input-dir", help="directory of the input dataset", required=True
    )
    parser.add_argument(
        "--output-dir", help="directory of the output dataset", required=True
    )
    parser.add_argument(
        "--logs-dir",
        help="directory of the logs",
        default=None,
    )
    parser.add_argument("--num-workers", help="number of workers", default=2, type=int)
    parser.add_argument(
        "--rehydrate",
        help="flag to rehydrate data (turn on for FineWeb2)",
        action="store_true",
    )

    args = parser.parse_args()

    main(args)
