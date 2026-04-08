import torch
import torch.nn as nn
import torchvision.models as models

class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()

        self.cnn = models.vgg16(pretrained=True)
        self.cnn.classifier = nn.Identity()
        for param in self.cnn.parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM(25088, 128, batch_first=True)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()

        x = x.view(batch_size * seq_len, C, H, W)
        features = self.cnn(x)

        features = features.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(features)

        out = self.fc(lstm_out[:, -1, :])
        return out