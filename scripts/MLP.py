import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


# Define the Simple MLP Model
class SimpleMLP(nn.Module):
    def __init__(self, input_dim1, input_dim2, num_classes):
        super(SimpleMLP, self).__init__()
        self.layer1 = nn.Linear(input_dim1 + input_dim2, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = F.relu(self.layer1(x))
        x = self.dropout(x)
        x = F.relu(self.layer2(x))
        x = self.dropout(x)
        x = F.softmax(self.layer3(x), dim=1)
        return x


# Updated train_model function
def train_model(dataset, epochs=100):
    loss = None
    possible_labels = ['plain nature', 'detailed nature', 'Agriculture', 'villages', 'city']

    entropies1 = [torch.tensor(d['entropies'], dtype=torch.float) for d in dataset]
    entropies2 = [torch.tensor(d['dwt'], dtype=torch.float) for d in dataset]
    labels = [possible_labels.index(d['label']) for d in dataset]

    entropies1 = torch.stack(entropies1)
    entropies2 = torch.stack(entropies2)
    labels = torch.tensor(labels, dtype=torch.long)

    train_data = TensorDataset(entropies1, entropies2, labels)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    num_classes = len(possible_labels)
    model = SimpleMLP(entropies1.shape[1], entropies2.shape[1], num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        for batch_idx, (data1, data2, target) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            output = model(data1, data2)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    return model


def predict(model, entropies, dwt):
    possible_labels = ['plain nature', 'detailed nature', 'Agriculture', 'villages', 'city']
    model.eval()
    with torch.no_grad():
        entropies = torch.tensor(entropies, dtype=torch.float).unsqueeze(0)
        dwt = torch.tensor(dwt, dtype=torch.float).unsqueeze(0)
        output_ = model(entropies, dwt)
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
        dataset.append({'entropies': entropies, 'dwt': dwt_entropies, 'label': entry['label']})

    i = int(test_part * len(dataset))
    test_set = dataset[-i:]
    dataset = dataset[:-i]

    trained_model = train_model(dataset, epochs=100)
    print('Model trained.')

    for test in test_set:
        stats['test_samples'] += 1
        predicted_label = predict(trained_model, test['entropies'], test['dwt'])
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
