# --- Bước 1: Cài đặt và thiết lập môi trường ---
# !pip install -qq faiss-cpu transformers pandas numpy scikit-learn tqdm

import pandas as pd
import numpy as np
import torch
import faiss
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import torch.nn.functional as F

# --- Bước 2: Đọc và chuẩn bị dữ liệu ---
# !gdown --id 1N7rk-kfnDFIGMeXOROVTjKh71gcgx-7R # Tải lại nếu cần
DATASET_PATH = "2cls_spam_text_cls.csv"
df = pd.read_csv(DATASET_PATH)
messages = df["Message"].values.tolist()
labels = df["Category"].values.tolist()

# --- Bước 3: Chuẩn bị mô hình Embedding ---
MODEL_NAME = "intfloat/multilingual-e5-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

# --- Bước 4: Vector hóa dữ liệu ---
def get_embeddings(texts, model, tokenizer, device, batch_size=32):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Tạo Embeddings"):
        batch_texts = texts[i:i+batch_size]
        # Thêm tiền tố "query: " hoặc "passage: " theo đề xuất của mô hình E5
        batch_texts_with_prefix = [f"passage: {text}" for text in batch_texts]
        
        batch_dict = tokenizer(batch_texts_with_prefix, max_length=512, padding=True, truncation=True, return_tensors="pt")
        batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
        
        with torch.no_grad():
            outputs = model(**batch_dict)
        
        batch_embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1) # Chuẩn hóa L2
        all_embeddings.append(batch_embeddings.cpu().numpy())
        
    return np.vstack(all_embeddings)

X_embeddings = get_embeddings(messages, model, tokenizer, device)

# Mã hóa nhãn và tạo metadata
le = LabelEncoder()
y = le.fit_transform(labels)
metadata = [{"index": i, "message": message, "label": label, "label_encoded": y[i]} 
            for i, (message, label) in enumerate(zip(messages, labels))]

# --- Bước 5: Xây dựng Cơ sở dữ liệu Vector và Chia dữ liệu ---
TEST_SIZE = 0.1
SEED = 42
train_indices, test_indices = train_test_split(range(len(messages)), test_size=TEST_SIZE, stratify=y, random_state=SEED)

X_train_emb = X_embeddings[train_indices]
X_test_emb = X_embeddings[test_indices]
train_metadata = [metadata[i] for i in train_indices]
test_metadata = [metadata[i] for i in test_indices]

embedding_dim = X_train_emb.shape[1]
index = faiss.IndexFlatIP(embedding_dim) # IP (Inner Product) tương đương với Cosine Similarity cho vector đã chuẩn hóa
index.add(X_train_emb.astype('float32'))

# --- Bước 6: Xây dựng Logic Phân loại và Đánh giá ---
def classify_with_knn(query_text, model, tokenizer, device, index, train_metadata, k=3):
    # Tạo embedding cho query
    query_embedding = get_embeddings([query_text], model, tokenizer, device)
    
    # Tìm kiếm k hàng xóm gần nhất
    scores, indices = index.search(query_embedding.astype('float32'), k)
    
    # Lấy nhãn và bỏ phiếu
    neighbor_labels = [train_metadata[i]['label'] for i in indices[0]]
    prediction = max(set(neighbor_labels), key=neighbor_labels.count) # Majority vote
    
    # Thu thập thông tin về hàng xóm
    neighbor_info = []
    for i in range(k):
      neighbor_idx = indices[0][i]
      neighbor_info.append({
          "score": scores[0][i],
          "label": train_metadata[neighbor_idx]["label"],
          "message": train_metadata[neighbor_idx]["message"]
      })
      
    return prediction, neighbor_info

# --- Bước 7: Kiểm thử Pipeline ---
test_input = "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward!"
prediction, neighbors = classify_with_knn(test_input, model, tokenizer, device, index, train_metadata, k=5)

print(f'Tin nhắn: "{test_input}"')
print(f'Dự đoán: {prediction.upper()}')
print("\nCác hàng xóm gần nhất:")
for i, neighbor in enumerate(neighbors):
    print(f"{i+1}. Label: {neighbor['label']} | Score: {neighbor['score']:.4f} | Message: {neighbor['message'][:80]}...")