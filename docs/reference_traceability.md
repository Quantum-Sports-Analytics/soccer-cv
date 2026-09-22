# Traçabilité — références utilisées par étape du pipeline

Source : *Computer Vision for Team Sports — Curated paper reference*, Quantum Sports Analytics, 3 septembre 2026 (`CV_Team_Sports_Paper_Reference.pdf`).
Codes d'entrée tels qu'ils apparaissent dans le PDF. Étapes numérotées comme dans `pipeline_architecture.png`.

**Couverture :** 31 des 34 entrées du PDF sont utilisées. Les 8 entrées de la liste prioritaire (section G) le sont toutes.

---

## Par étape du pipeline

| Étape | Entrée | Papier | Ce qui en est repris |
|---|---|---|---|
| 0 Décodage / structure du flux | — | — | aucune entrée ne traite la structure du broadcast ; conçu à partir des contraintes de production (voir « trous » plus bas) |
| 1 Points clés du terrain | C2 | Broadcast2Pitch / ++ | détection multi-tâches points + lignes + cercles comme front-end de la calibration |
| 2 Calibration | C2 | Broadcast2Pitch / ++ | homographie par optimisation image par image sur indices denses |
| 3 Détection | E1 | RF-DETR | détecteur par défaut : Apache-2.0, meilleur sur RF100-VL donc meilleur transfert au fine-tuning sur données propres |
| 3 Détection | E2 | YOLO26 | alternative plus rapide (AGPL-3.0) ; **tiling adaptatif attentif aux bords** plutôt que grille fixe |
| 3 Détection | D4 | SoccerNet 2026 results | monter la résolution d'entrée avant de grossir le backbone |
| 4 Association | A10 | Deep-EIoU + GTA | base du tracker (EIoU pour mouvement sportif irrégulier) ; GTA identifié comme **strictement hors-ligne** → contrainte n°1 du mode live |
| 4 Association | A4 | OA-SORT | correction du coût positionnel sous occlusion partielle (+2,08 HOTA / +3,05 IDF1 en moyenne sur 4 trackers) |
| 4 Association | A3 | McByte++ | compensation de mouvement caméra **conditionnelle** (seulement quand la caméra bouge) |
| 4 Association | A5 | CAMELTrack | voie de montée en gamme (association apprise) ; **mise en garde** : les keypoints de pose dégradent SportsMOT |
| 4 Association | A7 | MATR | point de comparaison online vs offline (72,2 vs 86,8 HOTA) qui justifie le choix tracking-by-detection |
| 4 Association | D1 | SportsMOT | les deux propriétés qui cassent l'association : vitesse variable, apparence quasi identique |
| 4 Association | F2 | Livestock review | voies alternatives : profondeur injectée dans l'association, prédicteur de mouvement transformer |
| 5 Banque ReID | C3 | KPR | descripteurs par parties **avec score de visibilité par partie** |
| 5 Banque ReID | B1 | TOTNet | pondération par visibilité transférée du ballon aux crops de joueurs |
| 5 Banque ReID | E3 | DINOv3 | conclusion : le backbone n'est pas le goulot ; les gains 2026 sont dans la couche d'association |
| 6 Masques sélectifs | A1 | Selective Mask Propagation | **le cœur de l'étape** : déclenchement sur la marge d'assignation, override seulement sur contradiction confiante (~7 % des fenêtres), coût amorti 0,154–0,171 s/image |
| 6 Masques sélectifs | A9 | SAM 3 / 3.1 | backbone de segmentation vidéo ; multiplexing d'objets pour le budget ; licence à revoir |
| 6 Masques sélectifs | A2 | McByte | parent conceptuel (masques propagés comme indice d'association) ; argument du « pas de réglage par vidéo » |
| 6 Masques sélectifs | F3 | TABE / amodal | complétion amodale admise **comme indice d'association uniquement**, jamais comme mesure |
| 7 Détecteur de ballon | B5 | TGMA-Net | priors de mouvement signés (Double-FD) ; 1,87 M paramètres à 173 FPS → point de départ déployable |
| 7 Détecteur de ballon | B3 | TrackNetV5 | même idée arrivée indépendamment (MDD, champs de polarité signés) |
| 7 Détecteur de ballon | B1 | TOTNet | agrégation temporelle 3D + **perte pondérée par la visibilité** ; métrique sur images totalement occultées |
| 7 Détecteur de ballon | B6 | BlurBall | modéliser le flou comme signal (la traînée encode la vitesse) au lieu de le supprimer |
| 7 Détecteur de ballon | F4 | Deblurring evidence | **ne pas débruiter** : le flou léger aide les trackers, le déflouage générique nuit |
| 7 Détecteur de ballon | B8 | RacketVision | transfert multi-sport (+19,2 % mAP tennis) ; la fusion de pose exige de la cross-attention |
| 8 Graphe de tracklets | A10 | GTA | association globale de tracklets comme modèle du recollage post-Tier A |
| 9 OCR numéro / équipe / rôle | F4 | Deblurring evidence | **sélection d'images clés** (2–3 crops lisibles) plutôt que restauration |
| 9 OCR numéro / équipe / rôle | A10 | GTA | OCR sur crops de torse guidés par la pose + classification d'équipe |
| 9 OCR numéro / équipe / rôle | E3 | DINOv3 | modes d'échec nommés : éclairage/ombres, confusion gardien/arbitre, amorçage en début de séquence |
| 10 Croyance d'identité | C4 | HMM identity-aware MOT | **formalisme adopté** : HMM sur le graphe de tracklets, émissions = identifications rares et incertaines |
| 10 Croyance d'identité | A8 | Expected Probability of Detection | l'occlusion entre dans le modèle de capteur : une non-détection prédite par la géométrie ne coûte rien |
| 10 Croyance d'identité | C1 | LTPI | **fonction de coût CSIS** (coûts d'erreur asymétriques) → l'abstention devient une sortie de premier rang |
| 10 Croyance d'identité | C2 | Broadcast2Pitch++ | fusion souple sur indices hétérogènes (apparence, mouvement, attributs d'identité) |
| 11 Ballon 3D | B2 | Physics-Based 3D Ball Trajectory | **l'étape entière** : segmentation aux contacts, objectif de reprojection, modèle à gravité ajustée gagnant en monoculaire, jeux de données 3D |
| 12 État en coordonnées terrain | F1 | OrganoidTracker 2.0 | probabilités d'erreur par lien, calibrées depuis l'écart au deuxième meilleur appariement |
| 12 État en coordonnées terrain | C5 | Athlete fatigue | traitement cinématique donnant vitesse et accélération temporellement cohérentes |
| 12 État en coordonnées terrain | D4 | SoccerNet 2026 results | **optimiser et évaluer en coordonnées terrain, pas en pixels** |
| 13 / 14 Rendu et exports | — | — | pas d'entrée applicable |

## Choix transverses

| Sujet | Entrées | Ce qui en est repris |
|---|---|---|
| Généralisation sans réglage par vidéo | A1, A2, A3, A4 | privilégier les composants **sans entraînement et plug-and-play** ; le réglage par séquence est nommé comme la faiblesse du tracking-by-detection |
| Protocole d'évaluation | D1, A7, A1 | fixer détecteur / données / online vs offline / post-traitement avant toute comparaison — les chiffres SportsMOT ne sont pas tous mesurés dans le même cadre |
| Métriques | C1, B1, D4 | CSIS pour l'identité ; précision stratifiée par visibilité pour le ballon ; erreur en mètres pour la position |
| Format d'annotation | F2 | **masques plutôt que boîtes** : vérité terrain non ambiguë sous occlusion |
| Données de développement | D5, D2, C1, B2 | TeamTrack + PFF FC (sans NDA) ; FOOTPASS pour actions/rôles (CC BY-NC, vidéo sous NDA) ; LTPI pour l'identité ; B2 pour le ballon 3D |
| Licences | E1, E2, A9, E3, D2 | registre des obligations : Apache-2.0 vs AGPL-3.0 vs licence SAM vs accord Meta vs CC BY-NC |
| Emprunts hors sport | F1, F2, F3, C4 | quantification d'incertitude (microscopie), identité longue durée sur individus quasi identiques (élevage) |

## Entrées non utilisées, et pourquoi

| Entrée | Papier | Raison |
|---|---|---|
| A6 | SAMIDARE (SAM2MOT scènes denses) | redondant avec A1 pour l'étape 6 : régénération de masques selon la densité et mémoire sélective visent le même échec. À garder comme variante à comparer si A1 déçoit en mêlée. |
| B4 | TrackNetV6 | non évaluable — payant, sans code ni preprint. Le document lui-même recommande de ne pas le poursuivre et de comparer B3/B5 sur nos images. |
| D3 | SoccerNet-GAR | porte sur la reconnaissance d'activité de groupe, c'est-à-dire la couche **en aval** du pipeline. Son résultat (les positions battent les pixels de 9 points avec 438× moins de paramètres) est un argument fort pour construire la sémantique tactique sur les sorties de l'étape 12 — mais hors périmètre de ce document. |

## Trous dans la référence, comblés par conception

Trois parties du pipeline ne s'appuient sur aucune entrée, parce que la littérature référencée travaille sur des clips déjà découpés en un seul plan caméra :

- **Étape 0** — détection de coupures, de ralentis et de type de caméra, profil de broadcast (letterbox, graphiques, cadence).
- **Contrainte de cardinalité du roster** (étape 10) — au plus 11 par équipe, un gardien, remplacements monotones. Aucune entrée ne l'exploite ; c'est l'indice le plus informatif et le moins cher du problème d'identité.
- **Clause d'inobservabilité** (§1.2 du document) — séparer mesures / état du roster / estimations. Aucune entrée ne la formule, et c'est pourtant ce qui distingue une sortie honnête d'une extrapolation.
