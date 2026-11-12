import torch
from sentence_transformers import CrossEncoder

model = CrossEncoder(
    "Alibaba-NLP/gte-multilingual-reranker-base",
    trust_remote_code=True,
    max_length=512,
    device="cpu",
    model_kwargs={
        "dtype": torch.float32
    },
)

# Получаем настоящий Embedding слой GTE
embedding = model.model.new.embeddings.word_embeddings

print("Embedding shape:", embedding.weight.shape)
print("Embedding dtype:", embedding.weight.dtype)

# Перемещаем только Embedding на GPU
embedding = embedding.to("cuda")

print("Embedding moved to GPU")

# Проверяем индексацию
for seq_len in (32, 512):

    print(f"\nTesting sequence length: {seq_len}")

    input_ids = torch.randint(
        low=0,
        high=embedding.num_embeddings,
        size=(3, seq_len),
        device="cuda",
        dtype=torch.long,
    )

    output = embedding(input_ids)

    torch.cuda.synchronize()

    print("Output shape:", output.shape)
    print("SUCCESS")