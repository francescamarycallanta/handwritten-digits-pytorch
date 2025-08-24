"""
Neural network models are here.
There is a simple MLP and a small CNN.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTMLP(nn.Module):
      """A basic fully connected network.
      Good first model: fast and accurate.
      """
      def __init__(self, dropout_p: float = 0.3):
          super().__init__()
          self.flatten = nn.Flatten()
          self.fc1 = nn.Linear(28 * 28, 256)
          self.fc2 = nn.Linear(256, 128)
          self.fc3 = nn.Linear(128, 10)
          self.drop = nn.Dropout(dropout_p) # turns off random neurons to reduce overfitting

          # Kaiming init helps ReLU layers start in a good place
          for m in self.modules():
              if isinstance(m, nn.Linear):
              nn.init.kaiming_normal_(m.weight)
              nn.init.zeros_(m.bias)
            
      def forward(self, x):
          x = self.flatten(x) # 28x28 -> long vector
          x = F.relu(self.fc1(x)) # layer 1 + activation
          x = self.drop(x) # small regularization
          x = F.relu(self.fc2(x)) # layer 2 + activation
          x = self.drop(x)
          return self.fc3(x) # final scores for 10 digits    
                  
class MNISTCNN(nn.Module):
    """A tiny CNN sees small image patterns.
    Usually a bit more accurate on pictures.
    """
    def __init__(self, dropout_p: float = 0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), # shrink to 14x14
            
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), # shrink to 7x7
        )
        self.drop = nn.Dropout(dropout_p)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 10),
        )
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x) # find edges and shapes
        x = self.drop(x) # regularize features
        return self.classifier(x)
