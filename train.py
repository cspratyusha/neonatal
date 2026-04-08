import os
import torch
from torch.utils.data import DataLoader
from dataset import IcopeDataset
from model import CNN_LSTM
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("models", exist_ok=True)

dataset = IcopeDataset("dataset_frames")
print("Total samples:", len(dataset))

loader = DataLoader(dataset, batch_size=2, shuffle=True)

model = CNN_LSTM().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

epochs = 1
losses = []

for epoch in range(epochs):
    model.train()
    running_loss = 0

    all_preds = []
    all_labels = []

    for batch_idx, (inputs, labels) in enumerate(loader):
        print(f"Processing batch {batch_idx}")

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    if len(all_labels) > 0:
        acc = accuracy_score(all_labels, all_preds)
    else:
        acc = 0

    print(f"Epoch {epoch+1} Loss: {running_loss:.4f} Accuracy: {acc:.4f}")
    losses.append(running_loss)

torch.save(model.state_dict(), "models/best_model.pth")

plt.plot(losses)
plt.title("Training Loss")
plt.show()
