# scripts/02_embed.py
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# 1. Завантажити датасет із файлу data/arxiv_subset.parquet
input_path = "../data/arxiv_subset.parquet"
df = pd.read_parquet(input_path)

# 2. Підготувати тексти для кодування
# Переконуємося, що немає пропущених значень (NaN), замінюючи їх на порожні рядки за потреби
titles = df['title'].fillna('')
abstracts = df['abstract'].fillna('')

texts = (titles + " [SEP] " + abstracts).tolist()

# 3. Згенерувати ембеддинги текстів за допомогою моделі allenai/specter2_base з бібліотеки sentence-transformers
model = SentenceTransformer('allenai/specter2_base')

# 4. Закодувати всі тексти в ембеддинги з урахуванням вимог

embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)

# 5. Вивести в консоль метрики
total_texts = len(embeddings)
embedding_dim = embeddings.shape[1]
first_norm = np.linalg.norm(embeddings[0])

print(f"Загальна кількість оброблених текстів: {total_texts}")
print(f"Розмірність ембеддингів: {embedding_dim}")
print(f"Норма першого ембеддингу: {first_norm:.6f}")


# 6. Зберегти отримані ембеддинги у файл embeddings/embeddings.npy у форматі NumPy +
# 7. Перед збереженням переконатися, що директорія embeddings існує; за потреби створити її
output_dir = os.path.join("..", "embeddings")
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "embeddings.npy")
np.save(output_path, embeddings)
print(f"Ембеддинги успішно збережено у файл: {output_path}")

