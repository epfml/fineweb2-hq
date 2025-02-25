from datatrove.pipeline.base import PipelineStep, DocumentsPipeline


# Rehydrater from https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
class Rehydrater(PipelineStep):
    def run(
        self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1
    ) -> DocumentsPipeline:
        import bisect

        upsampling_weights = {1: 1, 2: 2, 3: 3, 5: 5, 100: 8, 1000: 1}
        # Sorted keys
        limits = sorted(upsampling_weights.keys())

        for doc in data:
            upsampling_weight = upsampling_weights[
                limits[
                    bisect.bisect_right(limits, doc.metadata["minhash_cluster_size"])
                    - 1
                ]
            ]
            # repeat each document upsampling_weight times
            for _ in range(upsampling_weight):
                yield doc
