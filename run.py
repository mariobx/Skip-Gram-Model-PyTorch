import datasets
from skipgram import check_cuda, Word2Vec_neg_sampling
import torch 
from torch.utils.data import DataLoader


data = datasets.word2vec_dataset(DATA_SOURCE = 'fordham-data', CONTEXT_SIZE=2, SUBSAMPLING=1e-5, SAMPLING_RATE=1e-3, FRACTION_DATA=1)
EMBEDDING_SIZE = 128
VOCAB_SIZE = len(data.word_to_ix)  # Vocabulary size based on the dataset
DEVICE = torch.device("cuda" if check_cuda() else "cpu")
BATCH_SIZE = 64
LEARNING_RATE = 0.001
model = Word2Vec_neg_sampling(embedding_size=EMBEDDING_SIZE, vocab_size=VOCAB_SIZE, device=DEVICE, negative_samples=10).to(DEVICE)
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


for epoch in range(5):  # Number of epochs
    total_loss = 0
    for batch in dataloader:
        input_word, context_word = batch
        input_word = input_word.to(DEVICE)
        context_word = context_word.to(DEVICE)

        # Forward pass
        loss = model(input_word, context_word)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")