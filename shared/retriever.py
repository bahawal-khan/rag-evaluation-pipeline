
import chromadb
from chromadb.utils import embedding_functions

from paths import CHROMA_DIR, COLLECTION_NAME


class Retriever:
    def __init__(self, top_k: int = 3):
        self.top_k = top_k
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)

    def retrieve(self, query: str) -> list[str]:
        results = self.collection.query(query_texts=[query], n_results=self.top_k)
        return results["documents"][0]


if __name__ == "__main__":
    r = Retriever(top_k=3)
    q = "Tier 3 Confidential data ko training mein kaise use kiya ja sakta hai?"
    for i, chunk in enumerate(r.retrieve(q)):
        print(f"[{i}] {chunk}\n")
