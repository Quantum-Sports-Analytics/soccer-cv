# Mise en ligne de la plateforme — à faire (≈ 20-25 min, surtout de l'attente)

État au 22/09 au soir : **la chaîne complète tourne sur GPU Cloud Run** (L4, europe-west1),
testée depuis le code de la plateforme en mode `BACKEND=cloudrun` :
upload → job GPU → identité → overlay → lecture vidéo. Il ne reste qu'à héberger la plateforme.

## 0. Pousser le code

```bash
git push
```

## 1. Construire l'image de la plateforme (≈ 15 min)

Seulement l'image `tier-bc` (plateforme). L'image GPU `tier-a` actuelle est **validée, on n'y touche pas**
(chaque job GPU télécharge de toute façon le code à jour depuis le bucket).

```bash
gcloud builds submit --config deploy/cloudbuild_platform.yaml --project quantum-analytics-495309
```

## 2. Déployer (≈ 2 min)

```bash
gcloud run deploy soccer-cv-demo \
  --image europe-west1-docker.pkg.dev/quantum-analytics-495309/soccer-cv/tier-bc:latest \
  --region europe-west1 --project quantum-analytics-495309 \
  --service-account soccer-cv-submitter@quantum-analytics-495309.iam.gserviceaccount.com \
  --memory 4Gi --cpu 2 --timeout 3600 \
  --max-instances 1 --min-instances 1 --session-affinity --no-cpu-throttling \
  --set-env-vars BACKEND=cloudrun \
  --no-allow-unauthenticated
```

## 3. Ouvrir

```bash
gcloud run services proxy soccer-cv-demo --region europe-west1 --project quantum-analytics-495309 --port 8080
```
puis **http://localhost:8080**. L'URL publique `https://soccer-cv-demo-….run.app` est privée
(authentification Google requise) ; pour la partager : `roles/run.invoker` sur le service.

Après la démo, pour ne pas payer l'instance CPU au repos :
```bash
gcloud run services update soccer-cv-demo --region europe-west1 --min-instances 0
```

## Ce qui se passe quand on envoie une vidéo

1. la vidéo est découpée en morceaux de 8 Mo et stockée dans `gs://soccer-cv/runs/<id>/` ;
2. la plateforme (CPU) découpe en plans caméra ;
3. **un Cloud Run Job GPU** est créé (1 L4 par plan) : détection, calibration, suivi, ReID, ballon ;
4. la plateforme fait l'identité et l'overlay, **supprime le job GPU**, et affiche le résultat.

L'indicateur en haut à droite montre le nombre de jobs GPU actifs ; le bouton *annuler* arrête le job.

## Temps mesurés (L4, clip de 40 s, 1 plan)

| Étape | Temps | / durée vidéo |
|---|---|---|
| démarrage GPU (1er job, image froide) | ~80-270 s | — |
| détection RF-DETR small 896 px | 40 s | ×1,0 |
| calibration terrain | **356 s** | **×8,9 — goulot** |
| suivi + ballon + résumés | 27 s | ×0,7 |
| ReID + résolution des croisements | 16 s | ×0,4 |
| identité + overlay (CPU plateforme) | 52 s | ×1,3 |

Un match de 90 min découpé en ~300-600 plans tourne en parallèle (jusqu'à 15 L4) :
la durée est gouvernée par le plus long plan, pas par le match.
La calibration (CPU pur) est la prochaine optimisation — voir `docs/IMPROVEMENTS.md`.

## Si quelque chose casse

- job GPU échoué : `gcloud logging read 'resource.type="cloud_run_job"' --limit 100 --project quantum-analytics-495309`
  (les plantages natifs affichent la pile Python grâce à `PYTHONFAULTHANDLER`) ;
- vérifier qu'aucun GPU ne tourne : `gcloud run jobs list --region europe-west1 | grep scv-`
  (un job `scv-*` n'existe que pendant un calcul).

## Plus tard (non bloquant)

- Reconstruire aussi l'image GPU (`gcloud builds submit --config deploy/cloudbuild.yaml`) pour passer
  à torch récent (index cu128). Le contournement actuel `PYTORCH_JIT=0` fonctionne ; à retester après rebuild.
