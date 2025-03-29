# RAG
LLM-Powered Booking Analytics &amp; QA System
# Hotel Booking Data Analytics & RAG-based Q&A

## Overview

This project processes hotel booking data to generate analytics and enables a Retrieval-Augmented Generation (RAG)-based question-answering system using FAISS and Sentence Transformers.

## Features

- **Data Preprocessing**: Cleans and prepares hotel booking data.
- **Analytics & Reporting**: Generates revenue trends, cancellation rates, geographical distribution, and booking lead time insights.
- **RAG-based Q&A**: Enables retrieval of relevant information using FAISS and Sentence Transformers.

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- faiss
- sentence-transformers

## Usage

### Execution Steps

1. **Install dependencies:**

   ```sh
   pip install pandas numpy matplotlib seaborn faiss-cpu sentence-transformers
   ```

2. **Clone the repository and navigate to the project folder:**

   ```sh
   git clone <repository_url>
   cd <project_folder>
   ```

3. **Run the script:**

   ```sh
   python main.py
   ```

4. **Load and preprocess data (if running interactively):**

   ```python
   data = load_and_preprocess_data("hotel_bookings.csv")
   ```

5. **Generate analytics:**

   ```python
   analytics_result = generate_analytics(data)
   print("Analytics:", analytics_result)
   ```

6. **Create a vector store for RAG-based retrieval:**

   ```python
   index, embeddings, texts = create_vector_store(data)
   ```

7. **Retrieve answers based on user queries:**

   ```python
   user_query = input("Enter your question: ")
   print("Answer:", retrieve_answer(user_query, index, texts))
   ```

##

