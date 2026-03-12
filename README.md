# 🖼️ Débruitage d'Images par Deep Learning

Un projet de vision par ordinateur visant à améliorer la qualité d'images corrompues par du bruit, en utilisant un réseau de neurones convolutionnel entraîné sur des données haute résolution.

---

## 🎯 Objectif

L'objectif de ce projet est de concevoir et d'entraîner un modèle d'intelligence artificielle capable de **supprimer le bruit présent dans des images numériques** afin d'en restaurer la qualité visuelle. Le modèle apprend à distinguer le signal utile du bruit parasitaire, et produit une version nettoyée de chaque image en entrée.

---

## 🧠 Modèle : DnCNN

Le modèle choisi est **DnCNN** (*Denoising Convolutional Neural Network*), introduit par Zhang et al. en 2017. Il s'agit d'une architecture de référence dans le domaine du débruitage d'images supervisé.

Son principe repose sur le **residual learning** : plutôt que de prédire directement l'image propre, le réseau apprend à estimer le bruit lui-même, qu'il soustrait ensuite à l'image corrompue. Cette approche simplifie considérablement la tâche d'apprentissage et améliore la vitesse de convergence.

---

## 📦 Dataset : DIV2K

Les données d'entraînement proviennent du dataset **DIV2K** (*DIVerse 2K resolution*), un benchmark reconnu dans la communauté de la restauration d'images. Il contient **800 images haute résolution** de grande qualité et de contenu varié — paysages, portraits, architectures, textures — ce qui favorise la généralisation du modèle.

Les images propres servent de *ground truth*, tandis que les versions bruitées sont générées dynamiquement lors de l'entraînement par ajout de **bruit gaussien synthétique** à intensité variable.

---

## ⚙️ Technologies utilisées

| Domaine | Outils |
|---|---|
| Langage | Python 3 |
| Deep Learning | PyTorch, torchvision |
| Traitement d'images | Pillow, OpenCV, scikit-image |
| Analyse & visualisation | NumPy, Matplotlib, Seaborn, Pandas |
| Environnement | Kaggle Notebooks, GPU T4×2 |

---

## 🔄 Pipeline du projet

Le projet est organisé en trois phases successives :

**1. Exploration des données**
Analyse approfondie du dataset DIV2K : vérification de la qualité, étude des résolutions, distribution des intensités, visualisation de la diversité des images via PCA et t-SNE.

**2. Entraînement du modèle**
Entraînement du réseau DnCNN avec génération dynamique de bruit gaussien, augmentations des données (flip, rotation, crop aléatoire), et optimisation via Adam avec un scheduler cosinus. Le gel partiel des premières couches (*layer freezing*) permet un entraînement efficace sur GPU T4×2.

**3. Évaluation des performances**
Mesure de la qualité de débruitage à l'aide de métriques standards — **PSNR**, **SSIM** et **MSE** — complétée par des analyses visuelles comparatives et des cartes d'erreur pour identifier les zones difficiles à restaurer.

---

## 📊 Métriques d'évaluation

- **PSNR** *(Peak Signal-to-Noise Ratio)* : mesure le rapport signal/bruit en décibels. Un gain de plusieurs dB par rapport à l'image bruitée reflète une restauration efficace.
- **SSIM** *(Structural Similarity Index)* : évalue la similarité perçue par l'œil humain en tenant compte des textures, des contrastes et de la luminosité.
- **MSE** *(Mean Squared Error)* : erreur quadratique moyenne entre l'image prédite et l'image originale.


