"""
Evaluation tools: test accuracy, confusion matrix, and pictures of mistakes.
Pictures help humans see where the model is confused.
"""


from typing import Tuple, List
import math
import numpy as np
import torch
import matplotlib.pyplot as plt

@torch.no_grad()
def test_and_confusion(model, loader, device) -> Tuple[float, torch.Tensor]:
    """Compute test accuracy and a 10x10 confusion matrix.
    Rows are true labels; columns are predictions.
    """
    model.eval()
    correct, total = 0, 0
    conf = torch.zeros(10, 10, dtype=torch.int64)

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    for t, p in zip(labels.view(-1), preds.view(-1)):
        conf[t.long(), p.long()] += 1
        
    return correct / total, conf

def plot_history(history: dict):
    """Draw training curves so you can see learning progress.
    Two plots: loss and accuracy.
    """
    plt.figure(figsize=(6,4))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.title("Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout(); plt.show()
    
    
    plt.figure(figsize=(6,4))
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["val_acc"], label="val_acc")
    plt.title("Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.tight_layout(); plt.show()

def show_misclassified(model, loader, device, max_items: int = 12):
    """Show a small grid of wrong predictions.
    Title says True label (T) and Predicted (P).
    """
    @torch.no_grad()
    def collect():
        images_list, labels_list, preds_list = [], [], []
        model.eval()
        for images, labels in loader:
            logits = model(images.to(device))
            preds = logits.argmax(1).cpu()
            mism = preds != labels
          if mism.any():
              wrong_images = images[mism]
              wrong_labels = labels[mism]
              wrong_preds = preds[mism]
              for img, y, p in zip(wrong_images, wrong_labels, wrong_preds):
                  images_list.append(img.squeeze(0))
                  labels_list.append(int(y.item()))
                  preds_list.append(int(p.item()))
                  if len(images_list) >= max_items:
                      return images_list, labels_list, preds_list
      return images_list, labels_list, preds_list

  imgs, ys, ps = collect()
  cols = 6
  rows = math.ceil(len(imgs)/cols) if imgs else 1
  plt.figure(figsize=(12, 2.2*rows))
  for i, (img, y, p) in enumerate(zip(imgs, ys, ps)):
      plt.subplot(rows, cols, i+1)
      img_show = img * 0.3081 + 0.1307 # undo normalization for display
      plt.imshow(img_show.numpy(), cmap='gray')
      plt.title(f"T:{y} / P:{p}")
      plt.axis('off')
  plt.suptitle("Misclassified samples")
  plt.tight_layout(); plt.show()
          
