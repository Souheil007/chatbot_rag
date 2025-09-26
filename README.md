# Generative AI – Insurance Chatbot with RAG

This RAG-ChatBot is a Python application designed to answer car-insurance questions using three reference PDF documents. It leverages **Retrieval-Augmented Generation (RAG)** to provide accurate, sourced responses based on the content of these documents.

## Summary

Build a retrieval-augmented conversational agent that:

* Accepts free-form questions.
* Returns accurate, document-specific answers.
* Displays sources for each answer.

The chatbot is built for Option B: Generative AI – Insurance Chatbot with RAG.

## Knowledge Base

The chatbot uses three main reference documents:

1. **MTPL Product Info (PDF)**
2. **User Regulations (PDF)**
3. **Terms & Conditions (PDF)**

## Deliverables

* A working end-to-end chatbot application using Streamlit.
* Demonstrations showing correct answers to document-specific queries.
* A brief framework for evaluating answer accuracy, relevance, and latency.

## Solution Outline

### Document Processing

* Extract and normalize text from PDFs.
* Chunk the text into passages with metadata (e.g., source, section headings).
* OCR-based chunking is performed using **Mistral OCR** for optimal text extraction.

### Embedding & Vector Store

* Generate vector embeddings for each text chunk.
* Store embeddings in **ChromaDB**, which persists on disk for offline access.
* ChromaDB enables semantic similarity search for retrieval.

### Retrieval Mechanism

* Implement **semantic similarity search** over the vector embeddings.
* Retrieve the top relevant chunks for a given query.
* Ranking and re-ranking methods were tested (MMR, hybrid, and RAG fusion), but pure semantic similarity yielded the best results.

### Response Generation

* Use **Google's Gemini Pro** model to synthesize responses based on the retrieved chunks and the user's query.
* Responses include source citations from the original documents.
* Chat history (up to 10 previous messages) is retained for contextual follow-up questions.

### Out-of-Scope Handling

* Detect unsupported queries and respond gracefully.
* Prevents the model from hallucinating answers unrelated to the uploaded documents.

### Evaluation Framework

* Test queries with known answers were defined.
* Evaluated using **precision, recall, F1-score**, and semantic similarity.
* Semantic similarity search proved most reliable for accuracy.

### User Interface & Experience

* Built with **Streamlit** for a clean, simple interface.
* Users can upload PDFs or use previously uploaded documents without reprocessing.
* The chat interface displays AI responses along with their source documents.
* Real-time processing with low latency.

## Installation & Setup

### Step 1: Create `.env` File

1. Create a new `.env` file in the project root.
2. Add your API keys:

```shell
GOOGLE_API_KEY="your_google_api_key_here"
MISTRAL_API_KEY="your_mistral_api_key_here"
```

* **Where to get your Google API key:** [Google AI Studio API Key](https://aistudio.google.com/app/apikey)
* **Where to get your Mistral API key:** Check your [Mistral account portal](https://console.mistral.ai/api-keys) after signing up.

### Step 2: Install Dependencies

```shell
pip install -r requirements.txt
```

### Step 3: Run the App

```shell
streamlit run app/app.py
```

* The app opens in your web browser.
* Upload your documents to start chatting.
* Press `CTRL+C` in the terminal to stop the app.

## Notes on Implementation

* **Mistral OCR** is used for text extraction and chunking.
* **Gemini Pro** is used for response generation.
* **ChromaDB** is used as the vector store for retrieval.
* Semantic similarity search consistently outperformed hybrid and RAG-fusion approaches.
* Session state and HyDE approaches were tested but removed due to worse performance.

## Visual Interface

* Initial launch: prompts for document upload.
* Subsequent launches: chat interface available with uploaded documents listed.
* AI responses include source information for transparency.

## Evaluation Approach

* Manual and automatic evaluation on document-specific queries.
* Metrics: semantic similarity, precision@k, F1-score.
* Focus on retrieving accurate answers from the documents and avoiding hallucinations.
