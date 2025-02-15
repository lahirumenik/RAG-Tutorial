
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import torch
import faiss #FAISS is the vector store

from llama_index.core import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.vector_stores.faiss import FaissVectorStore

documents = SimpleDirectoryReader("data/").load_data()
print(len(documents))

device  = "cuda" if torch.cuda.is_available() else "cpu"
d = 1024
faiss_index = faiss.IndexFlatL2(d)

#load the embedding model - model will be saved in the cache directory
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", device=device) 

vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context,
    embed_model = embed_model
)

index.storage_context.persist()

embeddings = embed_model.get_text_embedding("Hello World!")
print(len(embeddings))
print(embeddings[:5])