import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ContrastiveProjectionHead(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512, output_dim=128):
        super(ContrastiveProjectionHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        # L2 normalize the output embeddings to align on a hypersphere
        out = self.fc(x)
        return out / (torch.norm(out, p=2, dim=1, keepdim=True) + 1e-10)

class TripletDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.classes = np.unique(self.labels)
        
        # Group indices by class
        self.class_indices = {c: np.where(self.labels == c)[0] for c in self.classes}
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        anchor_feat = self.features[idx]
        anchor_label = self.labels[idx]
        
        # Sample positive (same class, different index if possible)
        pos_indices = self.class_indices[anchor_label]
        if len(pos_indices) > 1:
            # Filter anchor index
            valid_pos_indices = pos_indices[pos_indices != idx]
            if len(valid_pos_indices) > 0:
                pos_idx = np.random.choice(valid_pos_indices)
            else:
                pos_idx = idx
        else:
            pos_idx = idx
        pos_feat = self.features[pos_idx]
        
        # Sample negative (different class)
        neg_classes = self.classes[self.classes != anchor_label]
        neg_label = np.random.choice(neg_classes)
        neg_idx = np.random.choice(self.class_indices[neg_label])
        neg_feat = self.features[neg_idx]
        
        return (
            torch.tensor(anchor_feat, dtype=torch.float32),
            torch.tensor(pos_feat, dtype=torch.float32),
            torch.tensor(neg_feat, dtype=torch.float32)
        )

def train_contrastive_head(features, labels, epochs=20, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Contrastive Projection Head on {device}...")
    
    model = ContrastiveProjectionHead().to(device)
    dataset = TripletDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.TripletMarginLoss(margin=0.5, p=2)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for anchor, pos, neg in dataloader:
            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
            
            optimizer.zero_grad()
            emb_anchor = model(anchor)
            emb_pos = model(pos)
            emb_neg = model(neg)
            
            loss = criterion(emb_anchor, emb_pos, emb_neg)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(anchor)
            
        avg_loss = total_loss / len(dataset)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Triplet Loss: {avg_loss:.4f}")
            
    return model
