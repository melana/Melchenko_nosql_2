# scripts/04_search.py
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


# 1.Підключитися до індексу arxiv-papers у Pinecone і завантажити модель allenai/specter2_base
load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")  # для отримання повного abstract


# 2. Реалізувати функцію кодування запиту в ембеддинг
def encode_query(query_text: str) -> list:
    emb = model.encode(query_text, normalize_embeddings=True)
    return emb.tolist()

def print_results(title_section: str, results: list):
    print(f"\n{title_section}:")
    for idx, item in enumerate(results, start=1):
        print(f"{idx}. [{item['category']}] ({item['year']}) {item['title']}")
        print(f"   Abstract snippet: {item['abstract'][:100]}...") # обмежуємо виведення 100 символами  
        if 'score' in item:
            print(f"   Score/Distance: {item['score']:.4f}")



# 3. Виконати чистий семантичний пошук
query_1 = "teaching machines to recognize objects in pictures"

print(f"\nВиконуємо чистий семантичний пошук для запиту: '{query_1}'")

query_vector_1 = encode_query(query_1)

pinecone_res = index.query(
    vector=query_vector_1,
    top_k=TOP_K,
    include_metadata=True
)

clean_search_results = []
for match in pinecone_res['matches']:
    meta = match['metadata']
    clean_search_results.append({
        "title": meta['title'],
        "category": meta['category'],
        "year": int(meta['year']),
        "abstract": meta['abstract'],
        "score": match['score']
    })

print_results("Результати чистого семантичного пошуку (Pinecone)", clean_search_results)

# 4. Пошук з фільтрацією
query_filter = "reinforcement learning"
query_vector_filter = encode_query(query_filter)

# Приклад A: статті по reinforcement learning за останні 5 років і категорія cs.LG
res_a = index.query(
    vector=query_vector_filter,
    top_k=TOP_K,
    include_metadata=True,
    filter={
        "category": {"$eq": "cs.LG"},
        "year": {"$gte": 2021}
    }
)

results_a = [{
    "title": m['metadata']['title'], "category": m['metadata']['category'],
    "year": int(m['metadata']['year']), "abstract": m['metadata']['abstract'], "score": m['score']
} for m in res_a['matches']]

if results_a:
    print_results("Фільтр А: reinforcement learning за останні 5 років і категорія cs.LG)", results_a)
else:
    print("Фільтр А: Відсутні статті по reinforcement learning за останні 5 років і категорія cs.LG")

# Приклад B: більш старі статті (до 2015 року), будь-яка категорія
res_b = index.query(
    vector=query_vector_filter,
    top_k=TOP_K,
    include_metadata=True,
    filter={
        "year": {"$lte": 2015}
    }
)

results_b = [{
    "title": m['metadata']['title'], "category": m['metadata']['category'],
    "year": int(m['metadata']['year']), "abstract": m['metadata']['abstract'], "score": m['score']
} for m in res_b['matches']]

if results_b:
    print_results("Фільтр B: статті по reinforcement learning до 2015 року", results_b)
else:
    print("Фільтр B: Відсутні статті по reinforcement learning до 2015 року")



# 5. Порівняти різні метрики схожості на локальних ембеддингах

EMB_PATH = "embeddings/embeddings.npy"

X = np.load(EMB_PATH)  # Матриця ембеддингів (N, 768)
q = np.array(query_vector_1)  # Вектор запиту (768,)

# Обчислення метрик
# 1) Cosine Similarity
norm_X = np.linalg.norm(X, axis=1)
norm_q = np.linalg.norm(q)
cosine_scores = np.dot(X, q) / (norm_X * norm_q)

# 2) Dot Product
dot_scores = np.dot(X, q)

# 3) L2 Distance
l2_distances = np.linalg.norm(X - q, axis=1)

# Вивести топ-5 статей для кожної метрики
def get_top_k_local(scores, reverse=True):
    if reverse:
        top_indices = np.argsort(scores)[::-1][:TOP_K]
    else:
        top_indices = np.argsort(scores)[:TOP_K]
    
    local_res = []
    for idx in top_indices:
        row = df.iloc[idx]
        local_res.append({
            "title": row["title"],
            "category": row["category"],
            "year": int(row["year"]),
            "abstract": row["abstract"],
            "score": float(scores[idx])
        })
    return local_res

print_results("Пошук: COSINE SIMILARITY", get_top_k_local(cosine_scores, reverse=True))
print_results("Пошук: DOT PRODUCT", get_top_k_local(dot_scores, reverse=True))
print_results("Пошук: L2 DISTANCE", get_top_k_local(l2_distances, reverse=False))