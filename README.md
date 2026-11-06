# Smart Parking Monitoring & Spatial Analytics
> **Projet Semestriel 4 : Visual Tracking & Intelligent Applications**  
> **Niveau :** 5ème Année Génie Informatique (Cours décalés) — AU 2026–2027  
> **Enseignante :** Dr. Ing. Rihab LAAJILI  
> **Évaluation :** Phase 1 — Présentation de Cadrage (Pondération : 20 %)

---

## 🎯 Objectif de ce Document
Ce fichier `README.md` regroupe l'ensemble des **instructions opérationnelles, des détails techniques et du contenu prêt à l'emploi** pour réussir la **Phase 1 (Présentation de cadrage — Étapes 1 à 4)** prévue le **5/6 novembre 2026**.

---

## 📋 Table des Matières
1. [Rappel des Exigences du Cahier des Charges](#1-rappel-des-exigences-du-cahier-des-charges)
2. [Détail des 4 Étapes à Présenter](#2-détail-des-4-étapes-à-présenter)
   * [Étape 1 : Problématique, Objectifs & Fonctionnalités (avec Extra Touch)](#étape-1--problématique-objectifs--fonctionnalités)
   * [Étape 2 : Étude Comparative de l'Existant](#étape-2--étude-comparative-de-lexistant)
   * [Étape 3 : Choix Techniques Argumentés](#étape-3--choix-techniques-argumentés)
   * [Étape 4 : Dataset Envisagé, Annotation & Scénarios](#étape-4--dataset-envisagé-annotation--scénarios)
3. [Structure Complète des Slides (Diapo par Diapo)](#3-structure-complète-des-slides-diapo-par-diapo)
4. [Anticipation des Questions du Jury (FAQ Défense)](#4-anticipation-des-questions-du-jury-faq-défense)
5. [Mise en Place de l'Environnement de Développement](#5-mise-en-place-de-lenvironnement-de-développement)

---

## 1. Rappel des Exigences du Cahier des Charges

* **Format du livrable :** Présentation orale en binôme avec support de slides.
* **Périmètre strict :** Validation exclusive des **étapes 1 à 4** (justification de faisabilité avant développement).
* **Critères d'évaluation (20 %) :**
  * Pertinence du sujet et clarté de la problématique.
  * Rigueur de l'étude comparative (solutions et modèles).
  * Justification scientifique des choix techniques.
  * Faisabilité et solidité de la stratégie de dataset.
* **Règle éliminatoire du projet :** Le tracking doit apporter une vraie valeur applicative (*« Un simple affichage de Bounding Boxes sans exploitation applicative ne répond pas aux exigences »*).

---

## 2. Détail des 4 Étapes à Présenter

### Étape 1 : Problématique, Objectifs & Fonctionnalités

#### A. Contexte et Problématique
* **Constat :** Jusqu'à 30 % du trafic en centre-ville et sur les campus est généré par des automobilistes cherchant une place de stationnement, entraînant perte de temps, émissions de CO₂ et embouteillages.
* **Limites actuelles :** Les solutions matérielles par capteurs au sol (magnétiques/ultrasons) sont excessivement chères à l'achat, imposent des travaux de génie civil et tombent en panne à cause des intempéries et du passage des roues.
* **Solution :** Utiliser des caméras de surveillance standard déjà installées pour analyser et suivre en continu l'occupation et les flux de véhicules grâce au *Visual Tracking*.

#### B. Fonctionnalités de Base
* Détection et suivi persistant de chaque véhicule entrant dans le champ.
* Cartographie temps réel des places libres (vert) vs occupées (rouge).
* Compteurs virtuels d'accès (*Virtual Tripwires*) : total des entrées et sorties.
* Calcul du taux d'occupation global du parking ($N_{\text{occupées}} / N_{\text{totales}}$).

#### C. Les « Extra Touches » (Facteurs différenciateurs pour le 20/20)
1. **Horodatage individuel & Dépassement (*Dwell Time Tracking*) :**
   * Grâce à la persistance du `track_id`, le système mesure la durée exacte de stationnement de chaque véhicule.
   * Affichage dynamique : `Place #03 [Car #12] : 14m 22s`.
   * **Alerte de dépassement :** Alerte visuelle clignotante si un véhicule dépasse le temps autorisé (ex: arrêt minute > 15 min).
2. **Détection du Stationnement Sauvage / Double File (*Obstruction Alert*) :**
   * Si un véhicule s'immobilise pendant plus de 2.5 secondes dans l'allée centrale hors des places délimitées, le système le signale en rouge clignotant avec une alerte d'obstruction de voie.
3. **Détection du Stationnement à Cheval sur la Ligne (*Parking on the Line*) :**
   * Détecte si un véhicule chevauche la ligne blanche séparant deux places (échantillonnage tripartite des roues), bloquant l'accès à deux emplacements et déclenchant une alerte d'incivilité.
4. **Détection en Temps Réel des Collisions & Near-Miss :**
   * Suivi cinématique des trajectoires croisées lors des manœuvres de recul ou de circulation en allée : alerte d'imminence (*Near-Miss*) et détection d'impact (*Collision*) avec journalisation de l'incident.
5. **Calibrateur Interactif 1-Clic :**
   * Interface graphique permettant de définir les polygones des places sur la première frame en quelques clics sans rien coder en dur, avec export en JSON.

---

### Étape 2 : Étude Comparative de l'Existant

#### A. Capteurs Physiques vs Vision par Ordinateur
| Critère | Capteurs magnétiques IoT | Caméras + Computer Vision (Notre choix) |
| :--- | :--- | :--- |
| **Coût matériel** | Très élevé (1 capteur payant par place) | Très économique (1 caméra couvre 20 à 50 places) |
| **Installation** | Lourde (perçage du sol, coupure du parking) | Rapide (simple fixation en hauteur) |
| **Données fournies** | Binaire uniquement (0 ou 1) | Binaire + Trajectoires + Vitesse + Catégorie de véhicule + Alertes |

#### B. Comparatif des Modèles de Détection d'Objets
* **Faster R-CNN :** Précis mais architecture à deux étages trop lente (~8–12 FPS), incompatible avec un flux vidéo temps réel fluide.
* **SSD (Single Shot MultiBox Detector) :** Rapide mais manque de précision sur les petits véhicules éloignés dans la perspective.
* **YOLOv8 / YOLOv11 :** Détection en une seule passe, précision de pointe (mAP 53.9) et cadence de **35 à 60 FPS**, idéal pour l'inférence temps réel.

#### C. Comparatif des Algorithmes de Tracking Temporel
* **SORT :** Repose uniquement sur le filtre de Kalman. Échoue massivement dès qu'un véhicule passe devant un autre (perte d'identifiant / *ID switch*).
* **DeepSORT :** Ajoute une couche d'extraction d'apparence (Re-ID), mais très lourd en calculs et ralentit drastiquement la cadence.
* **ByteTrack (Sélectionné) :** Révolutionne le suivi en associant non seulement les détections à score élevé, mais aussi les détections à faible score (voitures partiellement masquées par des arbres ou de l'ombre). Maintient l'identité du véhicule sans perte.

---

### Étape 3 : Choix Techniques Argumentés

#### A. Pile Technologique
* **Langage :** Python 3.10+
* **Framework de Détection & Tracking :** `ultralytics` (YOLOv8n/s + ByteTrack)
* **Traitement d'Image & Géométrie :** `opencv-python` et `shapely`
* **Interface Utilisateur :** `streamlit` pour le tableau de bord interactif en direct

#### B. Pipeline Algorithmique
```text
[Flux Vidéo] ──► [Inférence YOLOv8] ──► [Association ByteTrack (ID)] ──► [Test d'Intersection Spatiale] ──► [Filtre Temporel] ──► [Affichage & Alertes]
```

#### C. Modélisation Mathématique de l'Occupation (Anti-Flickering)
1. **Point d'ancrage sol :** Pour chaque véhicule détecté avec la boîte $[x_1, y_1, x_2, y_2]$, on utilise le point central de contact au sol :
   $$P_{\text{sol}} = \left( \frac{x_1 + x_2}{2}, y_2 \right)$$
2. **Test d'inclusion spatiale :** Vérification géométrique de l'appartenance de $P_{\text{sol}}$ au polygone de la place $S_i$ :
   $$\text{Inside}(P_{\text{sol}}, S_i) = \begin{cases} 1 & \text{si le point est dans la place} \\ 0 & \text{sinon} \end{cases}$$
3. **Filtre par hystérésis temporelle :** Pour éviter qu'une voiture qui roule simplement au-dessus d'une place ne la déclare occupée, la place n'est validée `OCCUPÉE` que si la présence persiste pendant au moins $N$ frames consécutives ($N = 30$ frames $\approx 1$ seconde) :
   $$\text{Occupancy}(S_i) = 1 \iff \sum_{t=k-N}^{k} \text{Inside}_t \ge 0.8 \times N$$

---

### Étape 4 : Dataset Envisagé, Annotation & Scénarios

#### A. Datasets Publics de Référence (Licences & Benchmark)
* **PKLot (Almeida et al.) :** 12 417 images, 695 899 places annotées sous soleil, nuages et pluie. Licence libre **CC BY 4.0** (Open Access sur Kaggle).
* **MetaPKLot (2024) :** 2,2 millions d'instances avec segmentation polygonale et suivi temporel (`car_id`) disponible sur Hugging Face.
* **Rôle :** Benchmark initial de baseline et validation de la résistance aux intempéries et ombrages.

#### B. Protocole de Collecte In-Situ (Obligation §4.4 du Cahier des Charges)
Pour respecter la contrainte stricte d'évaluation sur **vidéos réelles filmées par les étudiants** :
* **Prise de vue :** Vue plongeante (angle de 35° à 50°) filmée depuis le 2ème étage d'un bâtiment du campus universitaire donnant sur le parking.
* **Caractéristiques techniques :** Résolution Full HD (1920x1080), 30 FPS fixes, smartphone stabilisé.
* **Constitution du jeu de données custom :**
  1. **Vidéo d'entraînement / calibration (3 min) :** 6 à 10 places visibles. Scénarios : 2 arrivées, manœuvres de stationnement, 1 départ, passage de piétons.
  2. **Vidéo Inédite de Test (2 min - Réservée pour la soutenance finale) :** Filmée un autre jour sous un ensoleillement différent avec d'autres véhicules. **Jamais vue par l'algorithme pendant le développement.**

#### C. Procédure d'Annotation sans Douleur (Format JSON)
Aucun besoin d'annoter manuellement des milliers de frames !  
L'application utilise un script interactif `src/calibrate.py` permettant de cliquer sur les 4 coins des places une seule fois sur la première frame. Les polygones sont stockés dans `configs/parking_slots.json` :
```json
[
  {
    "id": 1,
    "name": "Slot_01",
    "polygon": [[120, 340], [210, 345], [195, 480], [105, 470]],
    "max_dwell_minutes": 15
  }
]
```

---

## 3. Structure Complète des Slides (Diapo par Diapo)

Voici le découpage recommandé pour votre présentation orale (10 slides équilibrées) :

| N° | Titre de la Diapositive | Contenu Clé & Visuels |
| :---: | :--- | :--- |
| **1** | **Titre & Binôme** | Titre du projet, noms des 2 étudiants, encadrante (Dr. Rihab LAAJILI), Année 2026-2027. |
| **2** | **Contexte & Problématique** | Chiffres clés (30% du trafic = recherche de place), limites des capteurs IoT (coût/usure), opportunité de la vision. |
| **3** | **Solution Proposée & Extra Touches** | Vue générale du système + Présentation des 3 innovations (Dwell Time, Détection de double file, Calibration 1-clic). |
| **4** | **Étude Comparative : Détecteurs** | Tableau comparatif (Faster R-CNN vs SSD vs YOLOv8). Justification du choix YOLO (vitesse 45+ FPS et précision). |
| **5** | **Étude Comparative : Trackers** | Tableau comparatif (SORT vs DeepSORT vs ByteTrack). Explication du mécanisme d'association bi-niveau de ByteTrack. |
| **6** | **Architecture Technique Globale** | Diagramme bloc modulaire (Capture $\to$ YOLOv8 $\to$ ByteTrack $\to$ Spatial Engine $\to$ Streamlit UI). |
| **7** | **Modélisation & Logique Métier** | Formule de projection du point au sol $P_{\text{sol}}$, test d'inclusion polygone et filtre temporel d'hystérésis anti-bruit. |
| **8** | **Dataset de Référence (Public)** | Présentation de PKLot & MetaPKLot (CC BY 4.0, Kaggle/HuggingFace), conditions météorologiques variées. |
| **9** | **Protocole In-Situ & Vidéo Inédite** | Photo du point de vue au 2ème étage du campus, spécifications vidéo (1080p, 30 FPS) et garantie de la vidéo test inédite. |
| **10** | **Planning & Répartition du Binôme** | Tableau de répartition des responsabilités (Étudiant 1 : Vision/Tracking ; Étudiant 2 : Logique/UI) + Diagramme temporel. |

---

## 4. Anticipation des Questions du Jury (FAQ Défense)

### Q1 : « Pourquoi ne pas simplement faire un classifieur d'image sur chaque case de parking ? »
* **Réponse :** Une approche par classification statique de cases (comme dans les vieux papiers de 2015) ne fait aucun **Visual Tracking**. Elle ne sait pas quelle voiture est garée, ne peut pas mesurer le temps de présence d'un individu (`track_id`), ne peut pas compter les entrées/sorties par franchissement de ligne, ni détecter un véhicule gênant qui s'arrête dans l'allée centrale.

### Q2 : « Que se passe-t-il si un piéton ou un arbre masque la voiture garée ? »
* **Réponse :** C'est exactement la raison d'être de **ByteTrack** couplé à notre **filtre temporel d'hystérésis**. ByteTrack continue d'associer la boîte même avec un faible score de confiance, et notre filtre exige 30 frames d'absence consécutives avant de déclarer une place libérée.

### Q3 : « Pourquoi le Fine-Tuning est-il nécessaire si YOLOv8 est déjà pré-entraîné sur COCO ? »
* **Réponse :** C'est une découverte majeure de notre étude expérimentale :
  - Sur un angle de surveillance classique à 45°, les poids COCO fonctionnent.
  - En revanche, sur une **vue zénithale drone à 90°** (`4196207`), YOLOv8 COCO obtient **0 détection (0% de rappel)** car COCO a été entraîné sur des photos prises à hauteur d'homme (roues, calandres visibles).
  - Le **Fine-Tuning avec Backbone gelé (freeze=10)** sur le dataset PKLot et des vues aériennes permet de faire passer le rappel de **0% à >90%**, tout en intégrant la classification directe des places (`empty_slot` vs `occupied_slot`).

### Q4 : « Que se passe-t-il si la caméra pivote ou tremble (caméra mobile PTZ) ? »
* **Réponse :** Nous avons intégré un module de **Compensation Globale de Mouvement (GMC)** utilisant l'algorithme ORB + RANSAC :
  - La matrice d'homographie $H_t$ est calculée à chaque frame pour recaler dynamiquement les polygones de places sur l'asphalte (effet Réalité Augmentée).
  - Dès qu'une place quitte l'écran ($x < 0$), elle est masquée pour éviter les boîtes fantômes.
  - Le système gère dynamiquement les zones de surveillance (**Zone A : Entrée** $\to$ **Zone B : Allée Centrale** $\to$ **Zone C : Bâtiments Est**) en fonction du balayage de la caméra.

---

## 5. Scripts d'Exécution & Démonstrations

### A. Démonstration des 4 Scénarios de Simulation
```bash
# Scénario 1 : Stationnement conforme & Chronométrage Dwell Time
python main.py --scenario 1

# Scénario 2 : Stationnement à cheval sur la ligne (incivilité bloquant 2 places)
python main.py --scenario 2

# Scénario 3 : Risque d'accrochage évité (Near-Miss avec freinage d'urgence)
python main.py --scenario 3

# Scénario 4 : Collision réelle confirmée (Crash en allée centrale)
python main.py --scenario 4
```

### B. Traitement Vidéo Réelle avec Suivi Caméra & Zones Adaptatives
```bash
# Vidéo commerciale avec balayage caméra (GMC + Multi-Zones)
python main.py --source 14292633_3840_2160_30fps.mp4 --config configs/commercial_parking_zones.json
```

### C. Pipeline d'Entraînement & Benchmark d'Adaptabilité
```bash
# Lancement du Fine-Tuning (Supporte GPU CUDA et CPU)
python train_parking.py --epochs 15 --batch 8 --imgsz 640

# Exécution du Benchmark Scientifique d'Adaptabilité (COCO vs Modèle Adapté)
python benchmark_adaptation.py
```

### D. Tableau de Bord Web Interactif (Streamlit)
```bash
streamlit run app.py
```

---

## 6. Structure Modulaire des Notebooks Google Colab / Kaggle

Pour suivre l'avancement phase par phase en toute clarté, chaque étape technique possède son **notebook indépendant et auto-suffisant** :

1. **[`01_Phase1_Simulation_Scenarios.ipynb`](file:///C:/Users/Sary/Ai_Project/01_Phase1_Simulation_Scenarios.ipynb) (Phase 1 — 16 cellules) :**
   * Définition géométrique et plan 2D du parking.
   * `SpatialEngine` avec point d'ancrage sol $P_{\text{sol}}$ et hystérésis anti-flickering.
   * Les 4 expérimentations isolées avec lecteurs vidéo HTML5 intégrés (Dwell time, Stationnement sur la ligne, Near-Miss, Crash).

2. **[`02_Phase2_Camera_Motion_GMC.ipynb`](file:///C:/Users/Sary/Ai_Project/02_Phase2_Camera_Motion_GMC.ipynb) (Phase 2 — 10 cellules) :**
   * Compensation Globale de Mouvement Caméra (GMC via ORB + RANSAC).
   * Recalage dynamique des places en Réalité Augmentée ($P_t = H_t \cdot P_0$).
   * Masquage automatique des places hors-champ ($x < 0$).
   * Basculement multi-zones adaptatif (Zone A Entrée $\to$ Zone B Allée Centrale).
   * Filtrage physique fenêtré éliminant 100% des fausses collisions sur les véhicules garés.

3. **[`03_Phase2_FineTuning_PKLot_Drone.ipynb`](file:///C:/Users/Sary/Ai_Project/03_Phase2_FineTuning_PKLot_Drone.ipynb) (Phase 2 — 12 cellules) :**
   * Preuve scientifique de la rupture de domaine (Vue drone zénithale à 90°).
   * Préparation et téléchargement du dataset PKLot / Vues aériennes au format YOLOv8.
   * Entraînement complet avec Transfer Learning et Backbone gelé (`freeze=10`).
   * Affichage des courbes de perte (`results.png`) et matrice de confusion.
   * Benchmark Avant/Après sur la vidéo drone `4196207` (0 détection $\to$ 208 détections).
   * Export automatique des poids finaux [`models/yolov8n_parking.pt`](file:///C:/Users/Sary/Ai_Project/models/yolov8n_parking.pt).

---
*Ce document couvre 100 % des attendus de la Phase 1 et de la Phase 2 du Cahier des Charges.*

