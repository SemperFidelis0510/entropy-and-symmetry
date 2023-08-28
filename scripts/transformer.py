import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Define the Self-Attention Layer
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, queries):
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Split the embedding into self.head different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Scaled dot-product attention
        scores = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        scores = scores / (self.head_dim ** 0.5)
        attention = torch.nn.functional.softmax(scores, dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out


# Define the Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, 2 * embed_size),
            nn.ReLU(),
            nn.Linear(2 * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(0.1)
        self.batch_norm1 = nn.BatchNorm1d(embed_size)
        self.batch_norm2 = nn.BatchNorm1d(embed_size)

    def forward(self, value, key, query):
        attention = self.attention(value, key, query)

        # Reshape tensor before BatchNorm1d
        reshaped_attention = attention.view(-1, attention.size(-1))

        x = self.batch_norm1(self.norm1(reshaped_attention))
        x = x.view(attention.size())  # Reshape back to original size
        x = self.dropout(x)

        forward = self.feed_forward(x)

        # Reshape tensor before second BatchNorm1d
        reshaped_forward = forward.view(-1, forward.size(-1))

        out = self.batch_norm2(self.norm2(reshaped_forward))
        out = out.view(forward.size())  # Reshape back to original size

        return out


# Define the Model
class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.encoder = nn.Embedding(256, 512)  # Assuming the numbers are in the range 0-255
        self.transformer = TransformerBlock(embed_size=512, heads=8)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.transformer(x, x, x)
        x = self.classifier(x[:, 0, :])
        return x


def train_model(dataset, epochs=100):
    loss = None
    possible_labels = ['plain nature', 'detailed nature', 'Agriculture', 'villages', 'city']

    # Convert dataset to PyTorch tensors
    entropies = [torch.tensor(d['entropies'], dtype=torch.long) for d in dataset]
    labels = [possible_labels.index(d['label']) for d in dataset]
    entropies = torch.stack(entropies)
    labels = torch.tensor(labels, dtype=torch.long)

    # Create DataLoader
    train_data = TensorDataset(entropies, labels)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    # Initialize the model, loss, and optimizer
    num_classes = len(possible_labels)
    model = Classifier(num_classes=num_classes)

    # Xavier Initialization
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    criterion = nn.CrossEntropyLoss()

    # Initialize the model, loss, and optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # Training loop
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
        scheduler.step(loss.item())
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    return model


def predict(model, numbers):
    possible_labels = ['plain nature', 'detailed nature', 'Agriculture', 'villages', 'city']
    model.eval()
    with torch.no_grad():
        numbers = torch.tensor(numbers, dtype=torch.long).unsqueeze(0)
        output_ = model(numbers)
        predicted_label_idx = torch.argmax(output_, dim=1).item()
        return possible_labels[predicted_label_idx]


def main():
    path = "../entropy_results/m=2023-08-28_01-16-07/entropy_results.json"
    test_part = 0.1
    stats = {'test_samples': 0, 'right_predictions': 0}

    with open(path, 'r') as f:
        metadata = json.load(f)
    dataset = []

    for name, entry in metadata.items():
        dataset.append({'entropies': [s['result'] for s in entry['entropy_results']], 'label': entry['label']})

    i = int(test_part * len(dataset))
    test_set = dataset[-i:]
    dataset = dataset[:-i]

    trained_model = train_model(dataset, epochs=100)
    print('Model trained.')

    for test in test_set:
        stats['test_samples'] += 1
        predicted_label = predict(trained_model, test['entropies'])
        if predicted_label == test["label"]:
            stats['right_predictions'] += 1
            print(f'Predicted label: {predicted_label}.  Real label: {test["label"]}. Prediction correct!')
        else:
            print(f'Predicted label: {predicted_label}.  Real label: {test["label"]}. False prediction.')

    stats['success_rate'] = 100 * stats['right_predictions'] / stats['test_samples']
    print(f"{stats['right_predictions']} samples out of {stats['test_samples']} were predicted correctly.\n"
          f"The models success rate is: {stats['success_rate']}%")


if __name__ == "__main__":
    main()
