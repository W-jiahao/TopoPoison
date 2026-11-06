from .base import BaseEmbeddingModel, EmbeddingConfig


def _get_embedding_model_class(embedding_model_name: str = "nvidia/NV-Embed-v2"):
    if "GritLM" in embedding_model_name:
        from .GritLM import GritLMEmbeddingModel

        return GritLMEmbeddingModel
    if "NV-Embed-v2" in embedding_model_name:
        from .NVEmbedV2 import NVEmbedV2EmbeddingModel

        return NVEmbedV2EmbeddingModel
    if "contriever" in embedding_model_name.lower():
        from .Contriever import ContrieverModel

        return ContrieverModel
    if "text-embedding" in embedding_model_name:
        from .OpenAI import OpenAIEmbeddingModel

        return OpenAIEmbeddingModel
    if "cohere" in embedding_model_name:
        from .Cohere import CohereEmbeddingModel

        return CohereEmbeddingModel
    if embedding_model_name.startswith("Transformers/"):
        from .Transformers import TransformersEmbeddingModel

        return TransformersEmbeddingModel
    if embedding_model_name.startswith("VLLM/"):
        from .VLLM import VLLMEmbeddingModel

        return VLLMEmbeddingModel
    raise ValueError(f"Unknown embedding model name: {embedding_model_name}")