import torch
import torch.nn as nn
import torchvision.models as models


class CNN_LSTM(nn.Module):
    def __init__(self, hidden_size=256, num_layers=2, lstm_dropout=0.3, num_classes=2):
        super(CNN_LSTM, self).__init__()

        # ── Backbone: ResNet18 ──────────────────────────────────────────────
        # Much lighter than VGG16: 512-d output vs 25088-d.
        # Partial unfreeze: freeze stem + layer1-2, train layer3-4.
        base = models.resnet18(pretrained=True)

        frozen_prefixes = ("conv1", "bn1", "layer1", "layer2")
        for name, param in base.named_parameters():
            if any(name.startswith(p) for p in frozen_prefixes):
                param.requires_grad = False
            else:
                param.requires_grad = True   # layer3, layer4, fc ← trainable

        # Strip the final FC head — we only want spatial features
        self.cnn = nn.Sequential(*list(base.children())[:-1])  # → (B, 512, 1, 1)

        # ── Feature projection ──────────────────────────────────────────────
        # Squeeze 512 → 256 before LSTM so it's not overwhelmed
        self.proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(p=0.4),
        )

        # ── Temporal model: Bidirectional LSTM ─────────────────────────────
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
            bidirectional=True,             # forward + backward context
        )

        # ── Classifier head ─────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),   # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier init for linear layers, orthogonal for LSTM."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (batch, seq_len, C, H, W)
        B, T, C, H, W = x.size()

        # Per-frame CNN features
        x        = x.view(B * T, C, H, W)
        features = self.cnn(x)                    # (B*T, 512, 1, 1)
        features = features.view(B * T, 512)      # (B*T, 512)

        # Project + regularise
        features = self.proj(features)            # (B*T, 256)
        features = features.view(B, T, 256)       # (B, T, 256)

        # Temporal modelling
        lstm_out, _ = self.lstm(features)         # (B, T, hidden*2)

        # Use last timestep for classification
        out = self.classifier(lstm_out[:, -1, :]) # (B, num_classes)
        return out