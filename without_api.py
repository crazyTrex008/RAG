import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import faiss
from sentence_transformers import SentenceTransformer

# Step 1: Data Collection & Preprocessing
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df.fillna({"children": 0, "agent": 0, "company": 0}, inplace=True)
    df.drop_duplicates(inplace=True)
    return df

# Step 2: Analytics & Reporting
def generate_analytics(df):
    analytics = {}
    
    # Revenue trends
    df["total_revenue"] = df["adr"] * (df["stays_in_weekend_nights"] + df["stays_in_week_nights"])
    revenue_trends = df.groupby(["arrival_date_year", "arrival_date_month"])['total_revenue'].sum()
    
    # Cancellation rate
    cancellation_rate = (df["is_canceled"].sum() / len(df)) * 100
    
    # Geographical distribution
    geo_distribution = df["country"].value_counts()
    
    # Booking lead time
    lead_time_distribution = df["lead_time"].describe()
    
    analytics["revenue_trends"] = revenue_trends.to_dict()
    analytics["cancellation_rate"] = cancellation_rate
    analytics["geo_distribution"] = geo_distribution.to_dict()
    analytics["lead_time_distribution"] = lead_time_distribution.to_dict()
    
    return analytics

# Step 3: RAG-based Question Answering
model = SentenceTransformer("all-MiniLM-L6-v2")
def create_vector_store(df):
    texts = df.astype(str).agg(' '.join, axis=1).tolist()
    embeddings = model.encode(texts)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, embeddings, texts

def retrieve_answer(query, index, texts):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=1)
    return texts[I[0][0]]

# Main Execution
data = load_and_preprocess_data("hotel_bookings.csv")
index, embeddings, texts = create_vector_store(data)
analytics_result = generate_analytics(data)

# Example Usage
print("Analytics:", analytics_result)
user_query = input("Enter your question: ")
print("Answer:", retrieve_answer(user_query, index, texts))
