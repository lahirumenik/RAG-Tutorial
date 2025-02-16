# Chat With Your Own Documents
![image](https://github.com/user-attachments/assets/abccc18e-75c2-4acd-a1bd-46650aa2ea7e)

# RAG-Tutorial: Retrieval-Augmented Generation for Beginners


## Key Features of This Tutorial

✅ Uses **local models**, so no API keys are needed.  
✅ Can run on **CPU**, so no GPU is required.  
✅ Designed for **learning purposes** with simple implementation.  
✅ Covers **basic retrieval and generation** steps. 

## Introduction

This tutorial explains Retrieval-Augmented Generation (RAG) in an easy way. RAG helps AI find the right information before answering a question. Instead of guessing, the AI first searches through documents, picks the useful parts, and then gives a better answer. This makes the response more accurate and helpful.

### What are Embeddings?

Embeddings are a way to represent words or documents as numbers so that similar meanings have similar numbers.

For example:

- **"cat"** and **"kitten"** will have similar embeddings because they are related.
- **"car"** and **"vehicle"** will also have similar embeddings.
- But **"apple"** and **"car"** will have very different embeddings since they are unrelated.

This is useful in RAG because when you ask a question, the AI finds the most relevant text from a large document using embeddings before generating a response.

Imagine you have a **1000-page PDF** and you ask, an question Instead of reading all 1000 pages, the using embeddings, it is possible finds the sections that are related to your question and gives you an answer from it.

## What is a Vector Store?

A vector store is a specialized database designed to store and retrieve high-dimensional vectors efficiently. These vectors, also known as embeddings, are numerical representations of data (such as text, images, or audio) that capture semantic relationships. Unlike traditional databases that store raw text or structured data, vector stores allow for similarity searches, meaning they can quickly find items that are semantically related rather than just exact matches.For this tutorial we will use FAISS as the vector store. 

### Vector Store in This Tutorial

For this tutorial, a FAISS vector store has already been created using a document issued by the Central Bank of Sri Lanka. [https://www.cbsl.gov.lk/sites/default/files/cbslweb_documents/publications/aer/2023/en/Full_Text.pdf] 
You can also add your own document(s) to the `data` directory and run `vectorstore.py`, which will:

- Divide the document page by page.
- Create an embedding for each page.
- Store these embeddings in the FAISS vector store for efficient retrieval.

Note that there is an embedding model to convert text in to embeddings. you can observe that in `vectorstore.py` file


### How RAG Works

When you ask a question:

1. The system converts your question into a vector embedding (a numerical representation).
2. It searches the vector store to find the most relevant stored embeddings. (Use vector calculations to find similarities)
3. It retrieves the best-matching text and generates an answer using LLM
![image](https://github.com/user-attachments/assets/841af4ba-a027-405c-8a03-1eff546f71ca)

## How to Set Up and Run This RAG Tutorial

Follow these steps to install and run the project on your computer.

### 1. Clone the Repository

```bash
git clone https://github.com/lahirumenik/RAG-Tutorial.git
```

### 2. Navigate to the Project Folder

```bash
cd RAG-Tutorial
```

### 3. Create and Activate a Virtual Environment

#### Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux & macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Download a Small Language Model Locally

```bash
python download.py
```

*Note: This will download about **1GB** of data.*

### 6. Run the Application

```bash
python app.py
```

- After running this, it will show a **Gradio link** (e.g., `http://127.0.0.1:7860`)
- Open the link in your browser to interact with the RAG system.

### 7. Experiment with RAG

The main logic is in **rag.py**, which contains comments explaining the retrieval and generation process.
 

## Future Improvements

This is a simple RAG system. For better results, consider:

1. **Better embeddings** for improved retrieval.
2. **Using metadata** in search.
3. **Finetuning the LLM** for improved answers.
4. **Adjusting retrieval parameters** to get more relevant documents.
5. **Using cross-encoder reranking** for better selection.

This tutorial is designed to help you **understand RAG basics** and experiment with it on your own machine. Happy learning! 🚀


