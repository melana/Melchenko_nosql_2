# scripts/03_load_to_pinecone.py
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INPUT_PARQUET = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"
INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 768
BATCH_SIZE = 200   # Pinecone рекомендує батчі до 200 векторів

# Ініціалізація клієнта
if "PINECONE_API_KEY" not in os.environ:
    raise ValueError("Неможливо знайти PINECONE_API_KEY в змінних оточення або файлі .env")

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# 1. Створюємо індекс (якщо не існує) 
existing_indexes = [index.name for index in pc.list_indexes()]

if INDEX_NAME not in existing_indexes:
    print(f"Індекс '{INDEX_NAME}' не знайдено. Створення нового серверлес індексу...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="cosine",  
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print(f"Індекс '{INDEX_NAME}' успішно створено.")
else:
    print(f"Індекс '{INDEX_NAME}' вже існує. Підключення...")

index = pc.Index(INDEX_NAME)

# 2. Завантажити дані з файлів

df = pd.read_parquet(INPUT_PARQUET)
embeddings = np.load(INPUT_EMBEDDINGS)

if len(df) != len(embeddings):
    raise ValueError(f"Невідповідність розмірів: у паркеті {len(df)} записів, а в ембеддингах {len(embeddings)}!")

# 3. Підготувати дані для завантаження + 
# 4. Завантажити дані в Pinecone батчами і показувати прогрес.
total_records = len(df)
print(f"Початок завантаження {total_records} векторів в Pinecone батчами по {BATCH_SIZE}...")

# Використовуємо tqdm для відображення прогресу по батчах
for i in tqdm(range(0, total_records, BATCH_SIZE), desc="Завантаження в Pinecone"):
    batch_df = df.iloc[i : i + BATCH_SIZE]
    batch_emb = embeddings[i : i + BATCH_SIZE]
    
    vectors_to_upsert = []
    
    for local_idx, (_, row) in enumerate(batch_df.iterrows()):
        global_idx = i + local_idx
        paper_id = f"paper_{global_idx}"
        
        metadata = {
            "arxiv_id": str(row["id"]),
            "title": str(row["title"]),
            "abstract": str(row["abstract"])[:500],  # обмеження до 500 символів
            "authors": str(row["authors"])[:200],    # обмеження до 200 символів
            "year": int(row["year"]),
            "category": str(row["category"])
        }
        
        vectors_to_upsert.append({
            "id": paper_id,
            "values": batch_emb[local_idx].tolist(),
            "metadata": metadata
        })
    
    # Завантаження поточного батчу
    index.upsert(vectors=vectors_to_upsert)

# 5. Після завершення завантаження вивести в консоль загальну кількість векторів в індексі.

print("Завантаження успішно завершено!")

index_stats = index.describe_index_stats()
total_vectors = index_stats['total_vector_count']

print(f"Загальна кількість векторів в індексі '{INDEX_NAME}': {total_vectors}")