import streamlit as st
from collections import defaultdict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import os
from evaluation_dataset import coverage_test_data, mtpl_regulations, user_terms_conditions_filtered

def get_context_retriever_chain(vectordb,docs):
    """
    Create a context retriever chain for generating responses based on the chat history and vector database

    Parameters:
    - vectordb: Vector database used for context retrieval

    Returns:
    - retrieval_chain: Context retriever chain for generating responses
    """
    # Load environment variables (gets api keys for the models)
    load_dotenv()
    # Initialize the model, set the retreiver and prompt for the chatbot
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2, convert_system_message_to_human=True)
    
    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "lambda_mult": 0.7}  # lambda balances relevance/diversity
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a chatbot. You'll receive a prompt that includes a chat history and retrieved content from the vectorDB based on the user's question. Your task is to respond to the user's question using the information from the vectordb, relying as little as possible on your own knowledge. If for some reason you don't know the answer for the question, or the question cannot be answered because there's no context, ask the user for more details. Do not invent an answer. Answer the questions from this context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    # Create chain for generating responses and a retrieval chain
    chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retrieval_chain = create_retrieval_chain(retriever, chain)
    return retrieval_chain

def get_response(question, chat_history, vectordb,docs):
    """
    Generate a response to the user's question based on the chat history and vector database

    Parameters:
    - question (str): The user's question
    - chat_history (list): List of previous chat messages
    - vectordb: Vector database used for context retrieval

    Returns:
    - response: The generated response
    - context: The context associated with the response
    """
    chain = get_context_retriever_chain(vectordb,docs)
    response = chain.invoke({"input": question, "chat_history": chat_history})
    return response["answer"], response["context"]

def chat(chat_history, vectordb,docs):
    """
    Handle the chat functionality of the application

    Parameters:
    - chat_history (list): List of previous chat messages
    - vectordb: Vector database used for context retrieval

    Returns:
    - chat_history: Updated chat history
    """
    user_query = st.chat_input("Ask a question:")
    if user_query is not None and user_query != "":
        # Generate response based on user's query, chat history and vectorstore
        response, context = get_response(user_query, chat_history, vectordb,docs)
        # Update chat history. The model uses up to 10 previous messages to incorporate into the response
        chat_history = chat_history + [HumanMessage(content=user_query), AIMessage(content=response)]
        # Display source of the response on sidebar
        with st.sidebar:
            st.subheader("Retrieved Chunks with Distance(lower is better)")
            # Get documents along with similarity scores
            results = vectordb.similarity_search_with_score(user_query, k=5)

            # Deduplicate based on chunk content
            seen = set()
            unique_results = []
            for doc, score in results:
                if doc.page_content not in seen:
                    unique_results.append((doc, score))
                    seen.add(doc.page_content)

            # Display
            for doc, score in unique_results:
                st.markdown("-----------")
                st.write(f"Source: {doc.metadata.get('source','Unknown')}")
                st.write(f"Page: {doc.metadata.get('page','N/A')}")
                st.write(f"Chunk: {doc.page_content}") 
                st.write(f"Distance: {score:.4f}")
                    
    with st.sidebar:
        st.subheader("⚙️ Evaluation")

        # List uploaded documents
        upload_docs = os.listdir("docs")
        selected_doc_name = st.selectbox("Select document for evaluation", upload_docs)

        if st.button("Run Evaluation"):
            # Only proceed if the selected doc is one of the three allowed
            if selected_doc_name in [
                "Copy of mtpl_coverage.pdf",
                "Copy of mtpl_regulations.pdf",
                "Copy of user_terms_conditions_filtered.pdf"
            ]:
                # Map doc_name to its corresponding test data
                if selected_doc_name == "Copy of mtpl_coverage.pdf":
                    test_data = coverage_test_data
                elif selected_doc_name == "Copy of mtpl_regulations.pdf":
                    test_data = mtpl_regulations
                else:  # user_terms_conditions_filtered.pdf
                    test_data = user_terms_conditions_filtered

                # Call evaluation function
                evaluate_system(test_data, vectordb, docs, k=3)
            else:
                st.warning("Evaluation is only available for specific documents.")

    # Display chat history
    for message in chat_history:
            with st.chat_message("AI" if isinstance(message, AIMessage) else "Human"):
                st.write(message.content)
    return chat_history



import numpy as np
import time

# -----------------
# Evaluation helpers
# -----------------
def precision_at_k(retrieved, relevant, k=3):
    retrieved_k = retrieved[:k]
    return len(set(retrieved_k) & set(relevant)) / max(len(retrieved_k), 1)

def recall_at_k(retrieved, relevant, k=3):
    retrieved_k = retrieved[:k]
    return len(set(retrieved_k) & set(relevant)) / max(len(relevant), 1)

def mean_reciprocal_rank(retrieved, relevant):
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0

def exact_match(pred, gold):
    return int(pred.strip().lower() == gold.strip().lower())

def f1_score_answer(pred, gold):
    pred_tokens = set(pred.lower().split())
    gold_tokens = set(gold.lower().split())
    common = pred_tokens & gold_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def measure_response_time(chatbot_fn, query, chat_history, vectordb, docs):
    start = time.time()
    _ = chatbot_fn(query, chat_history, vectordb, docs)
    return time.time() - start


from sentence_transformers import util, SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
def semantic_similarity(pred, gold):
    return util.cos_sim(model.encode(pred), model.encode(gold)).item()

def evaluate_system(test_data,vectordb, docs, k=3):
    sims, times = [], []
    chat_history = []

    for item in test_data:
        query, gold = item["query"], item["gold_answer"]

        # --- Measure response time ---
        start = time.time()
        pred_answer, _ = get_response(query, chat_history, vectordb, docs)
        elapsed = time.time() - start
        times.append(elapsed)

        # --- Semantic similarity ---
        sim = semantic_similarity(pred_answer, gold)
        sims.append(sim)

        # --- Show per-question results ---
        st.markdown(f"**Q:** {query}")
        st.write(f"Predicted: {pred_answer}")
        st.write(f"Gold: {gold}")
        st.write(f"Semantic Similarity: {sim:.2f}, Response Time: {elapsed:.2f}s")
        st.markdown("---")

    # --- Aggregate results ---
    avg_sim = np.mean(sims)
    avg_time = np.mean(times)

    st.subheader("📊 Final Scores")
    st.write(f"Average Semantic Similarity: {avg_sim:.2f}")
    st.write(f"Average Response Time: {avg_time:.2f}s")

    print("----- Final Evaluation Scores -----")
    print(f"Average Semantic Similarity: {avg_sim:.2f}")
    print(f"Average Response Time: {avg_time:.2f}s")