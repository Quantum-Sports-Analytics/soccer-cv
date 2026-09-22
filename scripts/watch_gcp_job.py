#!/usr/bin/env python
"""Poll a Cloud Batch job until it is terminal; print state changes, then the task's
container log tail and the run manifest. Usage: watch_gcp_job.py <job_name> <gs://run> [max_minutes]"""
import json, re, sys, time
import requests, google.auth, google.auth.transport.requests

name, run = sys.argv[1], sys.argv[2].rstrip("/")
max_min = float(sys.argv[3]) if len(sys.argv) > 3 else 70
bucket, prefix = re.match(r"gs://([^/]+)/(.+)", run).groups()
creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])

def hdr():
    creds.refresh(google.auth.transport.requests.Request()); return {"Authorization": f"Bearer {creds.token}"}

def get(url, **kw):
    for _ in range(5):
        try:
            return requests.get(url, headers=hdr(), timeout=60, **kw)
        except requests.exceptions.RequestException:
            time.sleep(10)
    return None

t0, last = time.time(), None
while time.time() - t0 < max_min * 60:
    r = get(f"https://batch.googleapis.com/v1/{name}")
    st = (r.json() if r is not None else {}).get("status", {})
    state = st.get("state"); counts = st.get("taskGroups", {}).get("group0", {}).get("counts", {})
    if (state, str(counts)) != last:
        print(f"[{time.time()-t0:5.0f}s] {state} {counts}", flush=True); last = (state, str(counts))
    if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
        break
    time.sleep(30)
uid = (r.json() if r is not None else {}).get("uid", "")
body = {"resourceNames": ["projects/quantum-analytics-495309"], "orderBy": "timestamp desc", "pageSize": 60,
        "filter": f'resource.type="batch.googleapis.com/Job" AND labels.job_uid="{uid}" AND NOT textPayload:"installer.go" AND NOT textPayload:"nvidia"'}
lr = requests.post("https://logging.googleapis.com/v2/entries:list", headers=hdr(), json=body, timeout=60)
ents = lr.json().get("entries", []) if lr.status_code == 200 else []
print(f"--- last {len(ents)} log lines ---")
for e in reversed(ents):
    print(e.get("timestamp", "")[11:19], (e.get("textPayload") or json.dumps(e.get("jsonPayload", {})))[:300])
m = get(f"https://{bucket}.storage.googleapis.com/{prefix}/manifest.json")
if m is not None and m.status_code == 200:
    print("--- manifest ---")
    for s in json.loads(m.content)["stages"]:
        print(f"  {s['stage']:16s} {s['seconds']:8.1f}s  {json.dumps(s['metrics'])[:150]}")
print("final:", state)
