
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import (
    load_index_from_storage,
    StorageContext,
)

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import torch


device  = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", device=device) 

vector_store = FaissVectorStore.from_persist_dir("./storage")
storage_context = StorageContext.from_defaults(
    vector_store=vector_store, persist_dir="./storage"
)
index = load_index_from_storage(storage_context=storage_context, embed_model=embed_model)

#Retrieve 5 documents (pages) similar to the query
query_engine = index.as_retriever(embed_model=embed_model, similarity_top_k=5)
response = query_engine.retrieve("How VAT related to inflation in 2024?")

for i, doc in enumerate(response):
    print(f"Document {i+1}:\n{doc.text}\n")