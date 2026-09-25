
import docx
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

from paths import DOCX_PATH, CHROMA_DIR, COLLECTION_NAME

CHUNK_SIZE = 400
CHUNK_OVERLAP = 60


def read_docx_text(path) -> str:
    document = docx.Document(str(path))
    lines = []
    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style_name = para.style.name if para.style is not None else ""
        if style_name.startswith("Heading") or style_name == "Title":
            lines.append(f"\n## {text}\n")
        else:
            lines.append(text)
    return "\n".join(lines)


def chunk_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def store_chunks(chunks: list[str]):
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(name=COLLECTION_NAME, embedding_function=embed_fn)
    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        metadatas=[{"source": str(DOCX_PATH), "chunk_index": i} for i in range(len(chunks))],
    )
    return collection


if __name__ == "__main__":
    raw_text = read_docx_text(DOCX_PATH)
    chunks = chunk_text(raw_text)
    print(f"Total chunks created: {len(chunks)}\n")
    for i, c in enumerate(chunks):
        print(f"--- chunk {i} ---\n{c}\n")

    collection = store_chunks(chunks)
    print(f"\nStored {collection.count()} chunks in ChromaDB at '{CHROMA_DIR}' "
          f"(collection: '{COLLECTION_NAME}')")
