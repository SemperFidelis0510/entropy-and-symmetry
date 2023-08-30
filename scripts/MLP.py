import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


# Define the Simple MLP Model
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, dwt_input_dim, num_classes):
        super(SimpleMLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, num_classes)
        self.dwt_layer = nn.Linear(dwt_input_dim, 128) if dwt_input_dim else None
        self.dropout = nn.Dropout(0.5)

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


# Updated train_model function
def train_model(dataset, epochs=100):
    loss = None
    possible_labels = ['plain nature', 'detailed nature', 'Agriculture', 'villages', 'city']
    entropies = [torch.tensor(d['entropies'], dtype=torch.float) for d in dataset]
    dwt_entropies = [torch.tensor(d['dwt'], dtype=torch.float) if d['dwt'] else None for d in dataset]
    labels = [torch.tensor([1 if l in d['label'] else 0 for l in possible_labels], dtype=torch.float) for d in dataset]

    entropies = torch.stack(entropies)
    labels = torch.stack(labels)

    if any(dwt is not None for dwt in dwt_entropies):
        dwt_entropies = torch.stack([d if d is not None else torch.zeros_like(dwt_entropies[0]) for d in dwt_entropies])
        train_data = TensorDataset(entropies, dwt_entropies, labels)
    else:
        train_data = TensorDataset(entropies, labels)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    num_classes = len(possible_labels)
    input_dim = entropies.shape[1]
    dwt_input_dim = dwt_entropies[0].shape[0] if dwt_entropies[0] is not None else None

    model = SimpleMLP(input_dim, dwt_input_dim, num_classes)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        for batch_idx, batch in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            if len(batch) == 3:
                data, dwt_data, target = batch
                output = model(data, dwt_data)
            else:
                data, target = batch
                output = model(data)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    return model


def predict(model, entropies, dwt_entropies):
    possible_labels = ['plain nature', 'detailed nature', 'Agriculture', 'villages', 'city']
    model.eval()
    with torch.no_grad():
        entropies = torch.tensor(entropies, dtype=torch.float).unsqueeze(0)
        dwt_entropies = torch.tensor(dwt_entropies, dtype=torch.float).unsqueeze(0) if dwt_entropies else None
        output_ = model(entropies, dwt_entropies)
        predicted_label_idx = torch.argmax(output_, dim=1).item()
        return possible_labels[predicted_label_idx]


def main():
    path = "../entropy_results/m=2023-08-29_19-14-57/entropy_results.json"
    test_part = 0.1
    stats = {'test_samples': 0, 'right_predictions': 0}

    with open(path, 'r') as f:
        metadata = json.load(f)
    dataset = []

    for name, entry in metadata.items():
        entropies = [s['result'] for s in entry['entropy_results'] if s['method'] != 'dwt']
        dwt_entropies = next((s['result'] for s in entry['entropy_results'] if s['method'] == 'dwt'), None)
        labels = entry['label'] if isinstance(entry['label'], list) else [entry['label']]
        dataset.append({'entropies': entropies, 'dwt': dwt_entropies, 'label': labels})

    i = int(test_part * len(dataset))
    test_set = dataset[-i:]
    dataset = dataset[:-i]

    trained_model = train_model(dataset, epochs=100)
    print('Model trained.')

    for test in test_set:
        stats['test_samples'] += 1
        predicted_labels = predict(trained_model, test['entropies'], test['dwt'])
        correct = all(label in predicted_labels for label in test['label'])

        if correct:
            stats['right_predictions'] += 1
            print(f'Predicted labels: {predicted_labels}.  Real labels: {test["label"]}. Prediction correct!')
        else:
            print(f'Predicted labels: {predicted_labels}.  Real labels: {test["label"]}. False prediction.')

    stats['success_rate'] = 100 * stats['right_predictions'] / stats['test_samples']
    print(f"{stats['right_predictions']} samples out of {stats['test_samples']} were predicted correctly.\n"
          f"The model's success rate is: {stats['success_rate']}%")


if __name__ == "__main__":
    main()
