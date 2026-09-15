from langchain.embeddings import Embeddings
from fastembed import TextEmbedding

class FastEmbLangChainAdapter(Embeddings):
    def __init__(self, model_name: str):
        self.model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.embed(texts)
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]