# 🖼️ Restauration d'Images par Deep Learning

Un projet de vision par ordinateur visant à améliorer la qualité d'images dégradées — débruitage et super-résolution — en utilisant **SwinIR**, un réseau de neurones basé sur les Transformers, entraîné sur des données haute résolution.

---

## 🎯 Objectif

L'objectif de ce projet est de concevoir et déployer un pipeline complet de **restauration d'images** capable de :
- **Supprimer le bruit gaussien** présent dans des images numériques
- **Reconstruire les détails** perdus lors d'une dégradation

Le modèle apprend à distinguer le signal utile du bruit parasitaire, et produit une version restaurée de chaque image en entrée. Le projet inclut également une **application web** permettant d'utiliser le modèle directement depuis un navigateur.

---

## 🧠 Modèle : SwinIR

Le modèle utilisé est **SwinIR** (*Swin Transformer for Image Restoration*), publié par Microsoft Research en 2021. Il s'agit de l'état de l'art en restauration d'images, basé sur l'architecture **Swin Transformer**.

Son principe repose sur l'**attention par fenêtres** : plutôt que d'analyser toute l'image d'un coup (coûteux), le modèle découpe l'image en fenêtres locales de 8×8 pixels et calcule les relations entre pixels à l'intérieur de chaque fenêtre. Les fenêtres sont **décalées alternativement** à chaque couche pour permettre aux régions adjacentes d'interagir, capturant ainsi des structures à longue distance.

SwinIR surpasse les CNN classiques (comme DnCNN) grâce à sa capacité à modéliser des dépendances à longue portée tout en restant efficace en mémoire.

---

## 📦 Dataset : DIV2K

Les données d'entraînement proviennent du dataset **DIV2K** (*DIVerse 2K resolution*), un benchmark reconnu dans la communauté de la restauration d'images. Il contient **800 images haute résolution** de grande qualité et de contenu varié — paysages, portraits, architectures, textures — ce qui favorise la généralisation du modèle.

Les images propres servent de *ground truth*, tandis que les versions dégradées sont générées dynamiquement lors de l'entraînement :
- **Débruitage** : ajout de bruit gaussien synthétique (σ variable)
- **Super-résolution** : sous-échantillonnage bicubique ×4

---

## ⚙️ Technologies utilisées

| Domaine | Outils |
|---|---|
| Langage | Python 3 |
| Deep Learning | PyTorch, torchvision |
| Traitement d'images | Pillow, scikit-image |
| Analyse & visualisation | NumPy, Matplotlib, Seaborn, Pandas |
| Backend | Flask |
| Frontend | React, TypeScript, Tailwind CSS |
| Déploiement | Docker, Docker Compose |
| Environnement entraînement | Kaggle Notebooks, GPU T4×2 |

---

## 🔄 Pipeline du projet

Le projet est organisé en quatre phases successives :

**1. Exploration des données**
Analyse approfondie du dataset DIV2K : vérification de la qualité, étude des résolutions, distribution des intensités, visualisation de la diversité des images via PCA et t-SNE.

**2. Entraînement du modèle**
Entraînement de SwinIR sur des paires d'images dégradées/originales, avec augmentations des données (flip, rotation, crop aléatoire) et optimisation via Adam avec scheduler MultiStep. Les poids du meilleur modèle sont sauvegardés automatiquement à chaque amélioration du PSNR de validation.

**3. Évaluation des performances**
Mesure de la qualité de restauration à l'aide de métriques standards — **PSNR**, **SSIM** et **MSE** — complétée par des analyses visuelles comparatives et des comparaisons avec la baseline bicubique.

**4. Déploiement applicatif**
Application web complète conteneurisée avec Docker. L'utilisateur dépose une image dans l'interface, elle est traitée par SwinIR côté serveur, et le résultat est affiché avec un comparateur avant/après interactif.

---

## 📊 Métriques d'évaluation

- **PSNR** *(Peak Signal-to-Noise Ratio)* : mesure le rapport signal/bruit en décibels. SwinIR atteint ~32-33 dB sur DIV2K ×4, contre ~28 dB pour la baseline bicubique.
- **SSIM** *(Structural Similarity Index)* : évalue la similarité perçue par l'œil humain en tenant compte des textures, des contrastes et de la luminosité.
- **MSE** *(Mean Squared Error)* : erreur quadratique moyenne entre l'image restaurée et l'image originale.

---

## 🐳 Lancement rapide

```bash
# Définir le token HuggingFace (téléchargement des poids, une seule fois)
export HF_TOKEN=hf_votre_token

# Construire et lancer
docker compose build
docker compose up

# Ouvrir l'application
http://localhost:8000
```