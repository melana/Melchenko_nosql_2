# scripts/05_chunking.py
import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768
BATCH_SIZE = 100

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")


# 1.Вибрати 30 статей із найдовшими анотаціями.
df['abstract_len'] = df['abstract'].fillna('').str.len()
top_30_df = df.nlargest(30, 'abstract_len').copy()
print("Вибрано 30 статей з найдовшими анотаціями.")

# 2. Розбити тексти на чанки двома стратегіями:
# Fixed-size chunking
def fixed_size_chunking(text: str, chunk_size: int = 50, overlap: int = 15) -> list:
    words = text.split()
    chunks = []
    if not words:
        return chunks
    
    i = 0
    while i < len(words):
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += (chunk_size - overlap)
        # Перериваємо, якщо дійшли до кінця або залишився занадто малий хвіст
        if i >= len(words) or len(chunk_words) < chunk_size:
            break
            
    return chunks

# Semantic chunking
def semantic_chunking(text: str, max_words: int = 60) -> list:
    # Простий спліт за крапками/знаками з урахуванням скорочень
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_words = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_words = len(sentence.split())
        
        # Якщо додавання речення перевищить ліміт — закриваємо поточний чанк
        if current_words + sentence_words > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_words = sentence_words
        else:
            current_chunk.append(sentence)
            current_words += sentence_words
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks


# 3. Створити окремі індекси в Pinecone для кожного типу чанків 
indexes = {
    "fixed": "arxiv-chunks-fixed",
    "semantic": "arxiv-chunks-semantic"
}

existing_indexes = [index.name for index in pc.list_indexes()]

for key, idx_name in indexes.items():
    if idx_name not in existing_indexes:
        print(f"Створення індексу '{idx_name}'...")
        pc.create_index(
            name=idx_name,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    else:
        print(f"Індекс '{idx_name}' вже існує.")

index_fixed = pc.Index(indexes["fixed"])
index_semantic = pc.Index(indexes["semantic"])


# 4. Для кожного чанка: створити ембеддинг за допомогою моделі allenai/specter2_base та сформувати об’єкт +
# 5. Завантажувати чанки в Pinecone батчами і відображати прогрес.
def process_and_upload_chunks(strategy_name: str, pinecone_index):
    print(f"\nОбробка за допомогою стратегії: {strategy_name.upper()}")
    all_vectors = []
    
    for idx, row in top_30_df.iterrows():
        abstract_text = row['abstract']
        
        # Вибір стратегії
        if strategy_name == "fixed":
            chunks = fixed_size_chunking(abstract_text, chunk_size=50, overlap=15)
        else:
            chunks = semantic_chunking(abstract_text, max_words=60)
            
        for chunk_num, chunk_text in enumerate(chunks):
            chunk_id = f"chunk_{strategy_name}_{row['id']}_{chunk_num}"
            
            # Генерація ембеддингу
            emb = model.encode(chunk_text, normalize_embeddings=True).tolist()
            
            # Формування метаданих
            metadata = {
                "arxiv_id": str(row["id"]),
                "title": str(row["title"]),
                "chunk_text": str(chunk_text),
                "chunk_num": int(chunk_num),
                "year": int(row["year"]),
                "category": str(row["category"])
            }
            
            all_vectors.append({
                "id": chunk_id,
                "values": emb,
                "metadata": metadata
            })
            
    # Завантаження батчами
    print(f"Завантаження {len(all_vectors)} чанків в індекс...")
    for i in tqdm(range(0, len(all_vectors), BATCH_SIZE), desc=f"Upsert {strategy_name}"):
        batch = all_vectors[i : i + BATCH_SIZE]
        pinecone_index.upsert(vectors=batch)

# Виконуємо обробку та завантаження для обох стратегій
process_and_upload_chunks("fixed", index_fixed)
process_and_upload_chunks("semantic", index_semantic)


# 6. Реалізувати функцію пошуку по чанках
def search_chunks(query_text: str):
    print(f"\nПошук за запитом: '{query_text}'")
    
    query_vector = model.encode(query_text, normalize_embeddings=True).tolist()
    
    for strategy, idx_name in indexes.items():
        print(f"\nРезультати для індексу: {idx_name}")
        p_index = pc.Index(idx_name)
        
        res = p_index.query(
            vector=query_vector,
            top_k=5,
            include_metadata=True
        )
        
        for num, match in enumerate(res['matches'], start=1):
            meta = match['metadata']
            print(f"{num}. [Score: {match['score']:.4f}] {meta['title']}")
            print(f"   [Chunk #{meta['chunk_num']}] Text: {meta['chunk_text'][:100]}...")

# Тестові запити для перевірки
search_chunks("computational methods for nonlinear systems")
search_chunks("mathematical analysis and algebraic structures")