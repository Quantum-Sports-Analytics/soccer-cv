# Améliorations à apporter au modèle de base

État au 22 septembre 2026, après le premier passage complet sur un clip de 40 s
(Barça–Real, caméra principale, 1080p25, nuit sous projecteurs). Chaque entrée
indique **ce qui a été observé**, **ce qui est proposé** et **comment on saura
que c'est réglé**. Les priorités seront révisées après les premiers tests sur de
nouvelles vidéos depuis la plateforme — c'est la méthode retenue : la sortie
overlay est le juge, complétée par les métriques par vidéo affichées à côté.

Légende priorité : **P0** bloque la qualité de base · **P1** amélioration nette
attendue · **P2** utile, plus tard.

---

## Généralisation à d'autres vidéos (à vérifier en premier)

Le pipeline est conçu sans réglage par vidéo, mais cinq composants ont été
ajustés en regardant ce clip et doivent être confrontés à d'autres conditions
(jour, ombres dures, pluie/neige, pelouse jaunie, autres diffuseurs, 720p).

| # | Composant | Réglé sur ce clip | Risque | Comment on le verra |
|---|---|---|---|---|
| G1 | Masque de lignes blanches (`s2_calib.line_mask`) : top-hat > 28, S < 90, V > 120 | oui | soleil rasant, neige, lignes usées | fraction d'images calibrées et `err_px` par vidéo |
| G2 | Plage HSV du gazon (`teams.GRASS_LO/HI`, teinte 35–85) | oui | pelouse sombre / étalonnage TV | `off_pitch_rejected` anormal, équipes mélangées |
| G3 | Dimensions du terrain fixées à 105 × 68 m | oui | stades 100–105 × 64–68 | erreur systématique en mètres ; à libérer dans l'optimisation (une fois par match) |
| G4 | Caméra à roulis nul | oui | caméra épaule, drone, plans larges inclinés | `err_px` élevée malgré un masque propre |
| G5 | Seuils de décision des croisements (1,5 nat ; 0,03 cos = 1 nat) | 13 fenêtres validées | trop ou trop peu de `split` sur d'autres kits | ratio swap/split/keep par vidéo, revue des fenêtres |

---

## P0 — Bloquent la qualité de base

### I1. Calibration terrain — finir et brancher
- **Observé** : le masque de lignes est propre ; l'ajustement paramétrique converge à ~12 px quand le rond central est visible, tombe dans des minima locaux (~30 px) sur les vues de surface de réparation en partant de zéro.
- **Proposé** : (a) propagation depuis la meilleure image du plan vers l'avant et l'arrière — *en cours* ; (b) si insuffisant, exploiter la structure des lignes (segments LSD classés horizontaux/verticaux, ellipse du rond central) pour des correspondances explicites plutôt que le seul chamfer ; (c) libérer L × W par match (G3).
- **Branchements attendus** : filtre « pieds dans le terrain » en mètres dans `s4_track` (remplace le masque gazon ; règle le cas des coachs sur l'herbe derrière la touche), positions en mètres dans `s5_summarize`, seuils de croisement et de re-liaison en mètres, grille projetée dans l'overlay.
- **Réglé quand** : ≥ 95 % d'images calibrées sur la caméra principale, `err_px` médiane ≤ 6, plus aucune personne hors terrain dans l'overlay.

### I2. Détecteur de ballon dédié
- **Observé** : la classe `sports ball` du détecteur générique perd le ballon par intermittence (score 0,44 au mieux avec `small`, rien avec `medium`) ; 82 % vu, 15 % interpolé, ce qui est trop pour une mesure fiable.
- **Proposé** : détecteur léger à priors de mouvement signés (famille TGMA-Net / TrackNetV5, réf. B3 & B5 du document de conception), fenêtre temporelle de 3–5 images, entrée pleine résolution sur 2 crops autour de la prédiction. À entraîner sur des données publiques (jeux de B2) puis affiner.
- **Réglé quand** : ballon vu ou correctement déclaré occulté/hors champ sur ≥ 95 % des images.

### I3. Fragmentation des pistes pendant les mouvements de caméra
- **Observé** : 69 tracklets pour ~20 personnes sur 40 s ; nouvelles identités créées surtout quand un joueur quitte le champ ou lors des mouvements rapides. Sur ce clip la caméra bouge peu (2 px/image en médiane) ; ce sera pire sur d'autres plans.
- **Proposé** : (a) compensation de mouvement caméra *conditionnelle* (réf. A3) : homographie image-à-image estimée sur le fond, appliquée à la prédiction Kalman seulement quand la caméra bouge ; (b) re-liaison en mètres avec ensemble atteignable (v_max, a_max) pour exclure les candidats impossibles avant l'apparence ; (c) prolonger la durée de vie des pistes perdues en bord de champ.
- **Réglé quand** : nombre d'identités ≈ nombre de personnes distinctes vues ; taux de re-liaison correcte après sortie/retour mesuré sur les cas revus à la main.

---

## P1 — Amélioration nette attendue

### I4. ReID par parties avec visibilité
- **Observé** : OSNet-AIN générique distingue les coéquipiers avec une marge faible (écarts de cosinus 0,02–0,04) ; il suffit quand on moyenne beaucoup de crops, pas sur un croisement bref avec peu d'images propres.
- **Proposé** : modèle par parties (KPR, réf. C3) avec score de visibilité par partie, puis affinage sur des tracklets du domaine (auto-supervisé : les segments d'une même piste hors fenêtre ambiguë sont des positifs). Objectif : marges ≥ 0,08 entre coéquipiers.

### I5. Détecteur affiné football
- **Observé** : RF-DETR COCO détecte bien les joueurs (18,4/image) mais gardien et arbitres passent par le clustering couleur, qui est le mode d'échec classique.
- **Proposé** : affinage avec classes `player / goalkeeper / referee / ball` sur des données publiques ouvertes (TeamTrack, PFF FC) ; tiling adaptatif côté opposé pour les joueurs de 20–30 px.

### I6. Résolution des croisements — évidence plus riche
- **Observé** : une fenêtre est indécidable quand un participant a trop peu de crops (fragment de 0,7 s).
- **Proposé** : (a) élargir la fenêtre d'évidence dynamiquement jusqu'à N crops propres ; (b) inclure la pose (orientation du corps) comme indice de continuité ; (c) palier optionnel de propagation de masques (réf. A1, étape 6) déclenché uniquement sur les fenêtres `split`, puisqu'elles sont peu nombreuses (2 sur 40 s).

### I7. Overlay — lisibilité pour le jugement visuel
- **Proposé** : grille terrain projetée (calibration), trace des N dernières positions, marqueur distinct pour `split`/`swap`, mini-carte tactique en coin, tableau des fenêtres ambiguës cliquable dans la plateforme (aller directement au moment concerné).

---

## P2 — Plus tard

- **I8. Ballon 3D** — segmentation aux contacts + modèle de vol à gravité ajustée (réf. B2). Dépend de I1 et I2.
- **I9. Coupures et ralentis** — l'étape 0 est heuristique (histogramme HSV) ; un classifieur de type de plan (principale / gros plan / ralenti / graphiques) sera nécessaire sur des matchs entiers.
- **I10. Cardinalité nominative** — contrainte « ≤ 11 par équipe » aujourd'hui anonyme ; avec la feuille de match, les remplacements deviennent des évidences d'identité.
- **I11. Mode live** — Tier B en fenêtre glissante avec identités révisables (voir §3.4 du document de conception).
- **I12. Palier GPU** — moteurs TensorRT FP16 pour L4 ; mesurer le coût réel par étape (le budget actuel est estimé).

---

## Méthode de priorisation

1. Lancer chaque nouvelle vidéo depuis la plateforme ; lire d'abord les métriques (calibration, fenêtres, abstentions), puis l'overlay.
2. Classer chaque défaut observé dans une ligne ci-dessus (ou en ajouter une).
3. Ne pas régler un seuil sur une vidéo : chercher le changement qui améliore toutes les vidéos vues jusqu'ici — les runs sont rejouables en secondes à partir des détections sauvegardées.


## Mesuré sur GPU (22/09, L4 Cloud Run, clip 40 s)

- **P0 — Calibration = 356 s pour 40 s de vidéo (×8,9), goulot unique.** CPU pur (chamfer + Powell).
  Leviers : (a) une image clé toutes les 2 s au lieu de 1 s avec la passe 2 à position fixe (3 DOF),
  (b) paralléliser les images clés sur les 8 vCPU de la tâche, (c) grille grossière vectorisée en numpy / GPU,
  (d) à terme un modèle de points clés appris. *Terminé quand* : calibration ≤ ×1 la durée vidéo.
- **P1 — Démarrage à froid 80-270 s par job.** Image de ~10 Go. Leviers : image plus légère, streaming d'image
  Artifact Registry, ou un job par match plutôt que par vidéo courte.
- **P2 — Image GPU avec torch 2.6 (index cu124)** ; import de rfdetr plante sans `PYTORCH_JIT=0`.
  Dockerfile passé à cu128 ; à reconstruire et revalider.
