#!/usr/bin/env bash
# One-time GCP setup. Run from a machine where you are logged in as project owner.
set -euo pipefail
PROJECT=${PROJECT:-quantum-analytics-495309}
REGION=${REGION:-europe-west4}          # pick a region where you have L4 (g2) quota
BUCKET=${BUCKET:-${PROJECT}-soccer-cv}
SA_NAME=soccer-cv-runner

gcloud config set project "$PROJECT"
gcloud services enable batch.googleapis.com compute.googleapis.com storage.googleapis.com \
    artifactregistry.googleapis.com cloudbuild.googleapis.com run.googleapis.com logging.googleapis.com

gsutil ls -b "gs://$BUCKET" >/dev/null 2>&1 || gsutil mb -l "$REGION" "gs://$BUCKET"
gcloud artifacts repositories describe soccer-cv --location="$REGION" >/dev/null 2>&1 || \
  gcloud artifacts repositories create soccer-cv --repository-format=docker --location="$REGION"

# Runtime service account used BY the Batch VMs and Cloud Run
gcloud iam service-accounts describe "$SA_NAME@$PROJECT.iam.gserviceaccount.com" >/dev/null 2>&1 || \
  gcloud iam service-accounts create "$SA_NAME" --display-name="soccer-cv runtime"
SA="$SA_NAME@$PROJECT.iam.gserviceaccount.com"
for ROLE in roles/storage.objectAdmin roles/logging.logWriter roles/artifactregistry.reader roles/batch.agentReporter; do
  gcloud projects add-iam-policy-binding "$PROJECT" --member="serviceAccount:$SA" --role="$ROLE" --quiet >/dev/null
done

# Submitter service account: the key you register in the Claude Science workspace.
# Scoped to: submit Batch jobs, act as the runtime SA, read/write the bucket, read logs.
SUB=soccer-cv-submitter
gcloud iam service-accounts describe "$SUB@$PROJECT.iam.gserviceaccount.com" >/dev/null 2>&1 || \
  gcloud iam service-accounts create "$SUB" --display-name="soccer-cv submitter (agent)"
SUBSA="$SUB@$PROJECT.iam.gserviceaccount.com"
for ROLE in roles/batch.jobsEditor roles/logging.viewer; do
  gcloud projects add-iam-policy-binding "$PROJECT" --member="serviceAccount:$SUBSA" --role="$ROLE" --quiet >/dev/null
done
gcloud iam service-accounts add-iam-policy-binding "$SA" --member="serviceAccount:$SUBSA" --role=roles/iam.serviceAccountUser --quiet >/dev/null
gsutil iam ch "serviceAccount:$SUBSA:objectAdmin" "gs://$BUCKET"
gcloud iam service-accounts keys create submitter-key.json --iam-account="$SUBSA"
echo
echo "Done. Register submitter-key.json in Claude Science (Customize -> Credentials -> GCP), then delete the local file."
echo "Bucket: gs://$BUCKET   Region: $REGION   Runtime SA: $SA"
