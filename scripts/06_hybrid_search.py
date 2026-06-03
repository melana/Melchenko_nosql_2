# scripts/06_hybrid_search.py
import os
import math
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from typing import List, Tuple

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 10   # беремо ширше, щоб RRF міг переранжувати

# 2. Підключитися до Pinecone і використовувати модель allenai/specter2_base для векторного пошуку.
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet").reset_index(drop=True)


# 1. Побудувати локальний BM25-індекс за заголовками і анотаціями всіх статей.
corpus_text = (df['title'].fillna('') + " " + df['abstract'].fillna('')).str.lower().str.split()
bm25_idx = BM25Okapi(corpus_text.tolist())


# 3. Реалізувати Reciprocal Rank Fusion (RRF) для об’єднання ранжованих списків BM25 і векторного пошуку
def reciprocal_rank_fusion(
    rankings: List[List[int]],
    k: int = 60
) -> List[Tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# 4. Реалізувати функції пошуку:
# BM25
def search_bm25(query: str, top_k: int = TOP_K) -> List[int]:
    bm25_scores = bm25_idx.get_scores(query.lower().split())
    return list(np.argsort(bm25_scores)[::-1][:top_k])

# Векторний (Pinecone)
def search_vector(query: str, top_k: int = TOP_K) -> List[int]:
    query_embedding = model.encode(query, normalize_embeddings=True).tolist()
    res = index.query(vector=query_embedding, top_k=top_k, include_metadata=False)
    return [int(match['id'].split('_')[1]) for match in res['matches']]

# Гібридний (BM25 + векторний через RRF)
def hybrid_search(query: str, top_k: int = TOP_K) -> List[Tuple[int, float]]:
    bm25_ranking = search_bm25(query, top_k=top_k)
    vector_ranking = search_vector(query, top_k=top_k)

    return reciprocal_rank_fusion([vector_ranking, bm25_ranking])


# Допоміжна функція для відображення результатів
def display_results(method_name: str, results_list: list, is_hybrid: bool = False):
    print(f"\n   Метод: {method_name}")
    if not results_list:
        print("      Результатів немає")
        return
        
    for rank, item in enumerate(results_list[:5], start=1):
        if is_hybrid:
            doc_id, score = item
            score_str = f" [RRF={score:.4f}]"
        else:
            doc_id = item
            score_str = ""
            
        row = df.iloc[doc_id]
        title_truncated = row['title'] if len(row['title']) < 80 else row['title'][:77] + "..."
        print(f"      {rank}.{score_str} ({row['year']}) [{row['category']}] {title_truncated}")


# 5. Для демонстрації виконати три запити +
# 6. Вивести результати для кожного методу і порівняти:

queries = [
    "BERT fine-tuning",
    "Yann LeCun convolutional networks",
    "making computers understand human emotions from text"
]

for q_num, query in enumerate(queries, start=1):
   
    print(f"\nЗапит №{q_num}: '{query}'")
    
    bm25_res = search_bm25(query, top_k=TOP_K)
    vector_res = search_vector(query, top_k=TOP_K)
    hybrid_res = hybrid_search(query, top_k=TOP_K)
    
    display_results("Точний пошук BM25", bm25_res)
    display_results("Векторний пошук (Pinecone)", vector_res)
    display_results("Гібридний пошук", hybrid_res, is_hybrid=True)