from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter
from pyarrow.parquet import ParquetFile
from tqdm.contrib.concurrent import thread_map
import pyarrow as pa
from functools import partial
import numpy as np
import pandas as pd
from .utils import list_files


def load_training_embeddings(
    data_dir,
    num_samples,
    seed,
):
    dfs = []
    for file in list_files(data_dir):
        dfs.append(pd.read_parquet(file, columns=["metadata"]))
    embeddings = pd.concat(dfs)
    embeddings = embeddings[
        embeddings["metadata"].apply(lambda x: x["label"] == "positive")
    ]
    embeddings = (
        embeddings["metadata"]
        .apply(lambda x: x["embeddings"][0])
        .sample(num_samples, random_state=seed)
    )
    embeddings = np.array([embd for embd in embeddings])
    return embeddings


class EmbeddingCosineSimilarityFilter(BaseFilter):
    name = "COSINE SIMILARITY"
    type = "🖩 EMBEDDINGS FILTER"

    def __init__(
        self,
        threshold: float,
        batch_size: int,
        positive_embeddings: list[list[float]],
        embedding_key=lambda x: x.metadata["embeddings"][0],
        exclusion_writer: DiskWriter = None,
    ):
        super().__init__(batch_size=batch_size, exclusion_writer=exclusion_writer)
        import numpy as np

        positive_embeddings = np.array(positive_embeddings)
        self.positive_embeddings = (
            positive_embeddings
            / np.linalg.norm(positive_embeddings, 2, axis=1)[..., np.newaxis]
        ).T
        self.embedding_key = embedding_key
        self.threshold = threshold

    def filter(self, document):
        pass

    def filter_batch(self, batch):
        import numpy as np

        batch_embeddings = np.array(
            [self.embedding_key(document) for document in batch]
        )
        batch_embeddings = (
            batch_embeddings
            / np.linalg.norm(batch_embeddings, 2, axis=1)[..., np.newaxis]
        )
        similarities = batch_embeddings @ self.positive_embeddings
        max_sims = np.max(similarities, axis=1).flatten().tolist()
        for document, score in zip(batch, max_sims):
            document.metadata["quality_score"] = score
        return map(lambda x: x > self.threshold, max_sims)


def estimate_cosine_threshold(
    input_dir,
    training_data_embeddings,
    num_samples,
    retention_rate,
    embedding_key=lambda x: x["embeddings"][0],
    num_workers=16,
):
    files = list_files(input_dir)
    num_samples_per_file = num_samples // len(files)

    training_data_embeddings = (
        training_data_embeddings
        / np.linalg.norm(training_data_embeddings, 2, axis=1)[..., np.newaxis]
    )

    def estimate_score(file, training_data_embeddings, num_samples_per_file):
        pf = ParquetFile(file)
        pf = next(
            pf.iter_batches(batch_size=num_samples_per_file, columns=["metadata"])
        )
        df = pa.Table.from_batches([pf]).to_pandas()
        embeddings = np.array(
            [embd for embd in df["metadata"].apply(embedding_key)],
        )
        embeddings = embeddings / np.linalg.norm(embeddings, 2, axis=1)[..., np.newaxis]
        cosine_similarity_scores = embeddings @ training_data_embeddings.T
        return np.max(cosine_similarity_scores, axis=1).tolist()

    scores = thread_map(
        partial(
            estimate_score,
            training_data_embeddings=training_data_embeddings,
            num_samples_per_file=num_samples_per_file,
        ),
        files,
        max_workers=num_workers,
    )
    scores = [el for arr in scores for el in arr]

    return np.quantile(scores, 1 - retention_rate)
