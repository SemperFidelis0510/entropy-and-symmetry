import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

all_labels = ['plain nature', 'detailed nature', 'Agriculture', 'villages', 'city']


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


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, dwt_input_dim, num_classes):
        super(SimpleMLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, num_classes)
        self.dwt_layer = nn.Linear(dwt_input_dim, 128) if dwt_input_dim else None
        self.dropout = nn.Dropout(0.5)
        self.possible_labels = all_labels

    def forward(self, x, dwt_x=None):
        x = F.relu(self.layer1(x))
        x = self.dropout(x)
        if dwt_x is not None and self.dwt_layer:
            dwt_x = F.relu(self.dwt_layer(dwt_x))
            x += dwt_x
        x = F.relu(self.layer2(x))
        x = self.dropout(x)
        x = F.softmax(self.layer3(x), dim=1)
        return x

    def train_model(self, dataset, epochs=100):
        loss = None
        entropies = [torch.tensor(d['entropies'], dtype=torch.float) for d in dataset]
        labels = [torch.tensor([1 if l in d['label'] else 0 for l in self.possible_labels], dtype=torch.float) for d in
                  dataset]
        entropies = torch.stack(entropies)
        labels = torch.stack(labels)
        train_data = TensorDataset(entropies, labels)
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                self.train()
                optimizer.zero_grad()
                output = self(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    def predict(self, entropies, dwt=None):
        self.eval()
        with torch.no_grad():
            entropies = torch.tensor(entropies, dtype=torch.float).unsqueeze(0)
            if dwt is not None and self.dwt_layer:
                dwt = torch.tensor(dwt, dtype=torch.float).unsqueeze(0)
            else:
                dwt = None
            output_ = self(entropies, dwt)
            predicted_label_idx = torch.argmax(output_, dim=1).item()
            return self.possible_labels[predicted_label_idx]


class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.encoder1 = nn.Embedding(256, 512)
        self.encoder2 = nn.Embedding(256, 512)
        self.transformer1 = TransformerBlock(embed_size=512, heads=8)
        self.transformer2 = TransformerBlock(embed_size=512, heads=8)
        self.classifier = nn.Linear(1024, num_classes)
        self.possible_labels = all_labels

    def forward(self, x1, x2):
        x1 = self.encoder1(x1)
        x2 = self.encoder2(x2)
        x1 = self.transformer1(x1, x1, x1)
        x2 = self.transformer2(x2, x2, x2)
        # Padding x1 to match the sequence length of x2
        if x1.size(1) < x2.size(1):
            padding = torch.zeros(x1.size(0), x2.size(1) - x1.size(1), x1.size(2)).to(x1.device)
            x1 = torch.cat((x1, padding), dim=1)
        # Padding x2 to match the sequence length of x1
        elif x2.size(1) < x1.size(1):
            padding = torch.zeros(x2.size(0), x1.size(1) - x2.size(1), x2.size(2)).to(x2.device)
            x2 = torch.cat((x2, padding), dim=1)

        x = torch.cat((x1, x2), dim=2)
        x = self.classifier(x[:, 0, :])
        return x

    def train_model(self, dataset, epochs=100):
        loss = None
        entropies = [torch.tensor(d['entropies'], dtype=torch.long) for d in dataset]
        dwt_entropies = [torch.tensor(d['dwt'], dtype=torch.long) for d in dataset]
        labels = [self.possible_labels.index(d['label']) for d in dataset]
        entropies = torch.stack(entropies)
        dwt_entropies = torch.stack(dwt_entropies)
        labels = torch.tensor(labels, dtype=torch.long)
        train_data = TensorDataset(entropies, dwt_entropies, labels)
        train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

        for epoch in range(epochs):
            for batch_idx, (data1, data2, target) in enumerate(train_loader):
                self.train()
                optimizer.zero_grad()
                output = self(data1, data2)
                loss = criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)
                optimizer.step()
            scheduler.step(loss.item())
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    def predict(self, numbers1, numbers2):
        self.eval()
        with torch.no_grad():
            numbers1 = torch.tensor(numbers1, dtype=torch.long).unsqueeze(0)
            numbers2 = torch.tensor(numbers2, dtype=torch.long).unsqueeze(0)
            output_ = self(numbers1, numbers2)
            predicted_label_idx = torch.argmax(output_, dim=1).item()
            return self.possible_labels[predicted_label_idx]


def process_json(path, test_part):
    with open(path, 'r') as f:
        metadata = json.load(f)
    dataset = []

    for entry in metadata:
        entropies = [s['result'] for s in entry['entropy_results'] if s['method'] != 'dwt']
        dwt_entropies = next((s['result'][:9] for s in entry['entropy_results'] if s['method'] == 'dwt'), None)
        dataset.append({'entropies': entropies, 'dwt': dwt_entropies, 'label': entry['label']})

    i = int(test_part * len(dataset))
    test_set = dataset[-i:]
    dataset = dataset[:-i]

    input_dim = len(dataset[0]['entropies'])
    dwt_input_dim = len(dataset[0]['dwt'])
    num_classes = len(all_labels)

    return dataset, test_set, input_dim, dwt_input_dim, num_classes


def evaluate_model(model, test_set):
    stats = {'test_samples': 0, 'right_predictions': 0}
    for test in test_set:
        stats['test_samples'] += 1
        predicted_label = model.predict(test['entropies'], test['dwt'])
        if predicted_label == test["label"]:
            stats['right_predictions'] += 1
            print(f'Predicted label: {predicted_label}.  Real label: {test["label"]}. Prediction correct!')
        else:
            print(f'Predicted label: {predicted_label}.  Real label: {test["label"]}. False prediction.')

    stats['success_rate'] = 100 * stats['right_predictions'] / stats['test_samples']
    print(f"{stats['right_predictions']} samples out of {stats['test_samples']} were predicted correctly.\n"
          f"The model's success rate is: {stats['success_rate']}%")


def main(model_type="SimpleMLP"):
    path = "../processed/results/123.json"
    test_part = 0.1

    dataset, test_set, input_dim, dwt_input_dim, num_classes = process_json(path, test_part)

    if model_type == "SimpleMLP":
        model = SimpleMLP(input_dim, dwt_input_dim, num_classes)
        model.train_model(dataset, epochs=100)
    elif model_type == "Transformer":
        model = Classifier(num_classes)
        model.train_model(dataset, epochs=100)
    else:
        print("Invalid model type")
        return

    print('Model trained.')
    evaluate_model(model, test_set)


if __name__ == "__main__":
    main(model_type="Transformer")  # Choose between "SimpleMLP" and "Transformer".
