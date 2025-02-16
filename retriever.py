
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import (
    load_index_from_storage,
    StorageContext,
    Document
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import torch
from llama_index.retrievers.bm25 import BM25Retriever

device  = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", device=device) 

splitter = SentenceSplitter(chunk_size=256)

vector_store = FaissVectorStore.from_persist_dir("./storage")
storage_context = StorageContext.from_defaults(
    vector_store=vector_store, persist_dir="./storage"
)
index = load_index_from_storage(storage_context=storage_context, embed_model=embed_model)

#Retrieve 5 documents (pages) similar to the query
query = "How VAT related to inflation in 2024?"
query_engine = index.as_retriever(embed_model=embed_model, similarity_top_k=3)
response = query_engine.retrieve(query)

retrieved_docs = ""
for i, doc in enumerate(response):
    retrieved_docs += response[i].get_content() + "\n"
    
splitter = SentenceSplitter(chunk_size=256)
nodes = splitter.get_nodes_from_documents(
    [Document(text=retrieved_docs)]
)


bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=2)
bm25_response = bm25_retriever.retrieve(query)

for i, doc in enumerate(bm25_response):
    print(f"Document {i+1}:\n{doc.text}\n")