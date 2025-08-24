# handwritten-digits-pytorch

Handwritten digit recognizer using **PyTorch**.  
Includes a simple MLP baseline and an optional CNN upgrade.  
Runs locally or in **Google Colab** (Colab badge below).

---

## 📦 Project Structure

├─ notebooks/
│ └─ mnist_pytorch_colab.ipynb # Colab-ready notebook
├─ src/
│ ├─ data.py # datasets, transforms, loaders
│ ├─ models.py # MLP + CNN architectures
│ ├─ train.py # train/val loops + early stopping
│ ├─ evaluate.py # test, confusion matrix, viz
│ └─ utils.py # seeding, metrics, helpers
├─ scripts/
│ ├─ train_mlp.py # CLI entry for MLP
│ └─ train_cnn.py # CLI entry for CNN
├─ tests/
│ └─ test_smoke.py # quick shape test
├─ requirements.txt
├─ README.md
├─ LICENSE
└─ .gitignore


---

## 🧠 What You’ll Learn

a) Feedforward networks (ReLU, Dropout), cross-entropy loss
b) Data pipelines, normalization, train/val split
c) Early stopping on validation loss
d) Evaluation: accuracy, confusion matrix, error analysis

## 📊 Expected Results

| Model                  | Params | Test Acc | Notes           |
| ---------------------- | -----: | -------: | --------------- |
| MLP (256→128, p=0.3)   | \~270k |    \~98% | Fast baseline   |
| Small CNN (32/64 + FC) | \~1.1M |    \~99% | Better accuracy |

## 🖼️ Visualisations

a) Training curves (loss & accuracy)
b) Confusion matrix heatmap (10×10)
c)Misclassified samples grid (title shows T: true / P: predicted)
