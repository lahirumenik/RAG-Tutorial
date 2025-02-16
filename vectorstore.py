"""
Created by Lahiru Menikdiwela
Date: 16 February 2025
"""


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

#!!!!!!Read the documents and create a vector store
#vector mean a represenation of documents  it is useful to identify if two documents are similar to each other using cosine similarity


documents = SimpleDirectoryReader("data/").load_data()
print(len(documents)) #Notice that this is equal to number of pages in the document 

device  = "cuda" if torch.cuda.is_available() else "cpu"
d = 1024 #no of dimension of embedding vector
faiss_index = faiss.IndexFlatL2(d)

#load the embedding model - model will be saved in the cache directory
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", device=device) 

#FAISS is a vector store there are many more vector stores like chroma, milvus, qdrant etc. 
vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context,
    embed_model = embed_model
)

index.storage_context.persist()

