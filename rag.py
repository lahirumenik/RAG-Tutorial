"""
Created by Lahiru Menikdiwela
Date: 16 February 2025
"""

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
from transformers import pipeline,  AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import re

device  = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", device=device) 


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Retrieval !!!!!!!!!!!!!!!!!!1
vector_store = FaissVectorStore.from_persist_dir("./storage")
storage_context = StorageContext.from_defaults(
    vector_store=vector_store, persist_dir="./storage"
)
index = load_index_from_storage(storage_context=storage_context, embed_model=embed_model) # load the vectorstore 

#Load the LLM model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("./model")
model = AutoModelForCausalLM.from_pretrained("./model", device_map="auto", torch_dtype=torch.bfloat16,)


#Retrieve 5 documents (pages) similar to the query 
query = "How VAT related to inflation in 2024?"
query = "Explain the Impact of Weather Patterns and Climate Change on Inflation in Sri Lanka"
query = "In which period daily power cut has started in Sri Lnaka and waht is the reason for it?"
query_engine = index.as_retriever(embed_model=embed_model, similarity_top_k=5) #top_k is the number of documents to retrieve based on embeddings

def generate_query(query):
    response = query_engine.retrieve(query)

    retrieved_docs = ""
    for i, doc in enumerate(response):
        retrieved_docs += response[i].get_content() + "\n"
        
    splitter = SentenceSplitter(chunk_size=200)
    nodes = splitter.get_nodes_from_documents(
        [Document(text=retrieved_docs)]
    )


    #### This is a reranking step to get the best document and this part is optional 
    #bm25i used to rerank the documents based on the keywords in the query
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=1)
    bm25_response = bm25_retriever.retrieve(query)

    # Get the best-ranked document
    best_doc = bm25_response[0]



    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Generation !!!!!!!!!!!!!!!!!!1


    llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)


    prompt = f"""
    You are an AI assistant. You have given the input document and a query. Answer to the question may be found in the document.
    If the user input is greeting provide appropraite response
    Provide a meaningful answer.:

    Document: {best_doc.text}

    Question: {query}

    Answer:
    """

    output = llm_pipeline(prompt, max_new_tokens=200, temperature=0.6, do_sample = True, top_k = 20)
    match = re.search(r"Answer:\s*(.*)", output[0]['generated_text'], re.DOTALL)
    return str(match.group(1)).strip() # pay attention to Doucument part in the  prompt. It containt the section of the document that is relevant to the query.

# print(generate_query(query))
#!!!!!!!!!!Note that this code is only for leanring purposes and has used light weight embeddings and llms so that any one can run it on their local machine with CPUs . 
""" 
For high quality answer generation need to do improvements on both retrieval and generation steps.
Few possible improments are:
1. Choice of embeddings
2. Use metadata in search
3. Improve LLM Cababilities either by finetuning or using a better LLM
4.Change embedding based retreieval and keyword based reranking parameters
5. Use more advanced reranking techniques like cross encoder

"""