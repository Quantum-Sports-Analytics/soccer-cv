# Broadcast football perception — architecture and difficulties

**Scope.** Ball, outfield players, goalkeepers and match officials tracked with persistent identity, plus per-frame field calibration, from a single TV broadcast feed. Design document: what the pipeline is, what makes the problem hard, and what each difficulty is answered with.

**Design point (fixed with you).** Horizontal scale-out on **Google Cloud Platform** GPUs (§3.5). Best-performing components chosen now; licence obligations recorded in §7 for cleanup before any commercial release. **Milestone 1 is judged on quality, not latency** — the mask tier (step 6) and a long Tier B window are on by default; live is milestone 5, and the same stage graph serves both (§3.4). Scope on the roster: players visible to the camera are what matters, no requirement to estimate off-frame positions — but **a player who leaves the frame and re-enters must come back with the same identity** (§1.2, §4.2).

**Status of the numbers in this document.** Accuracy figures attributed to published work are theirs, cited by entry code from your reference (A1, B2, C1 …). Every number describing *our* pipeline — latency, throughput, accuracy targets — is a design target or an engineering estimate, explicitly labelled as such. Nothing here has been measured on your footage, because no footage has been processed yet. §5 defines the harness that replaces these estimates with measurements.

---

## 1. Requirements and output contract

### 1.1 Functional

| Output | Rate | Notes |
|---|---|---|
| Player / goalkeeper / referee boxes + masks | 25 Hz (source rate) | on-pitch only; bench, crowd, staff suppressed |
| Track identity (persistent within match) | 25 Hz | with confidence and an explicit `unknown` state |
| Team, role, jersey number | per tracklet | revisable; sparse evidence, see §4.2 |
| Ball 2D position + visibility state | 25 Hz (50 Hz if source is 50p) | states: visible / occluded / out-of-frame / airborne |
| Ball 3D trajectory | per flight segment | with uncertainty; §4.4 |
| Camera calibration (pitch ↔ image) | 25 Hz | plus a validity flag |
| Pitch-coordinate positions, speed, acceleration | 25 Hz | derived; only where calibration is valid |

### 1.2 The unobservability clause

This is the single most important line in the contract, and it needs to be agreed before anything is built.

A main tactical broadcast camera does not show 22 players. It shows the ball neighbourhood — typically 10–16 field players, with the far-side full-back and the defending goalkeeper off-frame for long stretches, and it cuts to close-ups and replays several times a minute. **No monocular pipeline can measure the position of a player who is not in the frame.** A system that reports 22 positions on every frame from one broadcast feed is extrapolating, and if it does not say so, every downstream metric built on it inherits a fabrication.

So the pipeline emits three distinct things and never blurs them:

1. **Measurements** — a detected, tracked, calibrated position. Carries a per-link error probability (§4.9).
2. **Roster state** — for each of the up-to-22 identities on the pitch: `visible` / `off-frame` / `uncertain`, plus last known position and time since last observation. This is the honest answer to "where are the 22 players".
3. **Estimates** (optional, off by default) — off-frame position priors from role and phase-of-play models, flagged `estimated`, never exported into the same field as a measurement.

Full 22-player state is a multi-camera or tactical-feed product. If that is the end goal, the cheapest path is a second feed (tactical / panoramic), and this architecture accepts one by adding a camera dimension to the tracklet graph in Tier B — the rest is unchanged.

**Decision (yours):** visible players are the priority and off-frame estimation is out of scope. Item 3 above is therefore not built. The requirement that *is* in scope, and is the hard one, is **re-entry identity**: a player who leaves the frame — or is lost behind a cut — and reappears seconds or minutes later must be re-attached to the same identity. That is exactly what the Tier B tracklet graph exists for (§2.7, §4.2), and it is the primary quality criterion for milestone 2. It is measured as identity consistency across re-entries — the fraction of re-appearances correctly re-linked — on top of the CSIS score, not as aggregate HOTA, which barely registers re-entry failures.

### 1.3 Non-functional

- **Throughput:** faster than real-time per match on modest hardware; linear scale-out across GPUs (§3.3).
- **Generalisation:** any broadcast, no per-video tuning, no per-match manual calibration (§4.7). This is a hard constraint on component choice, not a nice-to-have: it eliminates any method that needs per-sequence threshold tuning.
- **Determinism and resumability:** every stage is a pure function of its inputs plus a config hash, writes to content-addressed storage, and is independently re-runnable. Re-running identity resolution after a model change must not require re-decoding video.
- **Live-readiness:** no stage may depend on the *whole* match being finished. Stages that want future frames are allowed a bounded lookahead (§3.4).

---

## 2. Architecture

![Three-tier pipeline. Tier A is per-shot and causal; Tier B resolves identity globally over tracklet features; Tier C fuses, renders and exports. Dashed box = quality tier only; dashed lines = cross-stage feedback.]({{artifact:art_54a994b5-d3f3-4675-b59d-c99463a054f7}})

Three tiers, split by what they need and what they cost.

**Tier A — per-shot, causal, GPU-bound.** Everything that touches pixels. Runs on one shot (one continuous camera take) at a time, needs no information from outside that shot, and is therefore the unit of parallelism (§3.2).

**Tier B — match-global, tracklet features only, CPU.** Identity resolution and ball trajectory fitting. Operates on closed tracklets and their cue summaries — a few megabytes per match, no pixels — so it costs under 1% of the pipeline and can be re-run in seconds while iterating.

**Tier C — fusion, render, export.** Pitch-space smoothing, overlay video, exports, QA report.

### 2.1 Stage 0 — ingest and broadcast structure

GPU-decode, then classify every frame into `main tactical` / `close-up` / `replay` / `graphics` / `other`, and detect cuts. A small classifier at 5 Hz plus a cut detector is enough.

This stage is usually skipped in research pipelines because benchmark clips are pre-cut to a single continuous main-camera take. On real broadcast it is load-bearing:

- A cut invalidates tracker state, motion models and calibration simultaneously. Detecting it is what stops a tracker from linking a defender to a coach's face in a close-up.
- Replays re-show play that already happened. Fed into an unsuspecting pipeline they create duplicate events and impossible trajectories. Broadcast replays are usually delimited by branded wipes and are slow-motion — both detectable.
- Shots are the natural shard boundary (§3.2).

### 2.2 Stage 1 — detection

**RF-DETR** (E1) as the default: Apache-2.0, leads RF100-VL which is the benchmark that actually predicts fine-tuning behaviour on custom data, and fine-tunes well on small custom sets. **YOLO26** (E2) is the faster alternative and is what the SoccerNet SynLoc winner used for small distant athletes in 4K, but it is AGPL-3.0 (§7).

Two non-obvious choices:

- **Resolution over model size.** The consistent lesson across SoccerNet 2026 teams (D4) was that raising input resolution helped — 224p → 448p → 720p — more reliably than scaling the backbone. A far-side player is 20–30 px tall in a 1080p broadcast frame; that is a resolution problem, not a capacity problem.
- **Boundary-aware adaptive tiling**, not a fixed grid. A fixed tile grid cuts players in half at seams, exactly where a cluster of players near the far touchline tends to sit. Adaptive tiling expands crops so objects are not split (E2 note).

Classes: `player`, `goalkeeper`, `referee`, `assistant referee`, `other person`, `ball`. Goalkeeper and referee as separate classes rather than post-hoc kit clustering — they are the two systematic failure cases of team clustering (E3), and they are cheap to learn directly.

### 2.3 Stage 2 — field calibration

Multi-task keypoint + line + circle segmentation feeding optimisation-based homography estimation, per frame, as in Broadcast2Pitch (C2). Dense line and circle cues matter because a broadcast frame frequently contains only two or three canonical keypoints; lines and arcs carry the rest of the constraint.

Three additions that the literature entries treat as details and production treats as essential:

1. **Player masking before line fitting.** Players occlude lines and their boots generate spurious edges. Mask detections out first — a tracking → calibration feedback edge.
2. **Temporal regularisation on camera parameters, not on the homography.** Broadcast cameras pan, tilt and zoom from a fixed tripod. Parameterising by pan/tilt/zoom and smoothing in that space keeps the solution physically plausible; smoothing homography matrix entries does not.
3. **Pitch-model prior with per-match dimension fitting.** Pitch length varies 100–105 m across stadiums. Fit dimensions once per match from accumulated observations, then hold them fixed — this removes a slow systematic error from every pitch-space measurement.

Calibration then feeds back into detection: reject detections outside the pitch polygon (plus a touchline margin) or grossly inconsistent with the expected player pixel-height at that pitch location. This one gate removes most of the crowd, bench, steward and photographer false positives that no amount of detector training reliably suppresses.

### 2.4 Stage 3 — association

Tracking-by-detection with **Deep-EIoU** (A10) as the base: the expanded-IoU association is built for the irregular, variable-speed motion that defeats constant-velocity assumptions, and on SoccerTrack it took HOTA from 0.42 to 0.54 while cutting identity switches from 630 to 325 relative to ByteTrack association. The honest reason to pick tracking-by-detection over an end-to-end tracker is the gap in your own reference: 86.8 HOTA for tracking-by-detection plus refinement versus 72.2 for the strongest end-to-end model (A7) — though see §5 on comparability, since those are not the same setting.

Three modifications, all training-free, all chosen because they cost nothing to try and do not compromise §1.3 generalisation:

- **Occlusion-corrected positional cost** (A4). Under partial occlusion the detected box shrinks and shifts toward the visible part, so the position cost *lies* to the assignment step. Estimating occlusion state and correcting the positional cost recovered +2.08 HOTA / +3.05 IDF1 on average when integrated into four different trackers — the cheapest available win.
- **Conditional camera-motion compensation** (A3). Apply compensation when the camera actually moves rather than on every frame; broadcast alternates between static holds and fast pans, and always-on compensation injects noise during the holds.
- **Physical motion limits in pitch space.** Once calibration exists, gate associations on plausible speed and acceleration in metres, not pixels. A pixel-space gate is meaningless when a far-side player moves 3 px and a near-side player moves 30 px for the same physical displacement.

**Selective mask propagation** (A1) sits on top as an optional quality tier. The dispatch signal is the assignment margin in the Hungarian cost matrix — the gap between best and second-best assignment. When that margin is small, or a track disappears and reappears, a window opens and a video object segmentation model propagates masks through it; the base tracker's output is only changed if the segmentation confidently contradicts it, which happens in about 7% of opened windows. This is the right shape of solution (pay only on the hard fraction) but it is also the most expensive stage in the pipeline by a factor of three to six (§3.1), which is why it is a tier and not a default.

### 2.5 Stage 4 — appearance

Part-based re-identification with per-part visibility scores (**KPR**, C3) rather than a single global embedding. The visibility scores are the reason: they let a crop that is 60% occluded contribute only the parts that are actually visible, with a weight, instead of poisoning the appearance bank with a blended two-player embedding. The same logic as the visibility-weighted loss in TOTNet (B1), applied to players.

Appearance embeddings run at 5 Hz, not 25 Hz — appearance changes slowly and this cuts the cost by 5× — plus on demand when the association step is uncertain.

Team assignment is **relative, not absolute**: cluster embeddings within the match into 4 groups (two outfield kits, two goalkeepers) plus officials, rather than classifying against a fixed kit vocabulary. A fixed vocabulary cannot survive an unseen third kit, and generalisation to any broadcast (§1.3) forbids it. Cluster identity is then anchored to teams by side-of-pitch at kickoff and by the scoreboard if available.

### 2.6 Stage 5 — ball

A dedicated detector, not a class in the player detector. The ball is 8–15 px, moves up to 30 m/s, is motion-blurred, and is occluded or out of frame for a large fraction of the match; it shares almost no statistics with a 30–100 px player.

- **Signed motion priors** — independently arrived at by B3 (MDD) and B5 (Double-FD), a convergence your reference calls the strongest signal in the ball-tracking literature this cycle. An absolute frame difference discards direction; a signed difference field encodes the vector from the darkening departure region to the brightening arrival region. TGMA-Net (B5) reaches this at 1.87M parameters and 173 FPS, which makes it the deployment-grade starting point.
- **Temporal aggregation with a visibility-weighted loss** (B1) — evidence across a window rather than per frame, and no penalty for failing to detect a ball that is genuinely hidden, so the model learns to interpolate rather than to hallucinate. TOTNet reports occluded-frame accuracy 0.63 → 0.80 and RMSE 37.30 → 7.19 on its benchmarks.
- **An explicit ball state machine** — `visible`, `occluded`, `out-of-frame`, `in-flight`, `held`. Most ball-tracking failures in production are not localisation errors; they are a tracker confidently reporting a position for a ball that is in a player's hands at a throw-in, or off-camera entirely.
- **No deblurring** (F4). The evidence is unambiguous: light motion blur *improves* most trackers, deblurring helps only on severely blurred video and actively harms lightly blurred video, with ringing artifacts corrupting features. Model the blur as signal instead (B6): the streak encodes the velocity within the exposure, so joint blur-and-position estimation yields a free velocity measurement. Ordering: capture, then blur-as-signal, then task-aware restoration, and generic deblurring last if ever.

### 2.7 Stage 6 — identity resolution (Tier B)

Identity is not a per-frame classification problem. It is a global assignment over the tracklet graph under sparse, unreliable evidence — you see a jersey number occasionally, at low confidence, never during occlusion.

The formalism: a **hidden Markov model over the tracklet graph**, hidden states = true identities, emissions = the occasional uncertain identifications (C4 — which comes from livestock tracking, structurally our problem: visually near-identical individuals, dense mutual occlusion, long sequences, identity that must persist). Evidence entering the model:

- jersey-number OCR on **keyframe-selected** crops — pick the 2–3 legible crops from a 25–50 frame tracklet rather than restoring bad ones (F4);
- team and role cluster membership;
- part-based appearance similarity with visibility weights;
- motion and time feasibility (a player cannot be in two places, and cannot cross the pitch in 0.4 s);
- **roster cardinality** — at most 11 per team on the pitch, exactly one goalkeeper per team, substitutions monotone, red cards permanent. A hard constraint on the global assignment, and an unusually strong one: it converts identity from N independent guesses into a constrained matching where a confident assignment *excludes* competitors.

**Abstention is a first-class output.** From LTPI (C1): a wrong identity corrupts every event attributed to that trajectory until recovery, whereas an abstention costs far less — which is why their CSIS metric weights the two asymmetrically. We adopt that loss directly, and the system reports `unknown` rather than guessing when belief is below the CSIS-optimal threshold. The abstention rate is a reported QA number, not a hidden failure.

### 2.8 Stage 7 — ball 3D (Tier B)

Segment the trajectory at contact events, then fit a forward-simulated flight model per segment against a reprojection objective (B2). Model choice follows their benchmark result rather than intuition: with **monocular** input, the fitted-gravity model wins on every soccer setting, while richer spin-decomposition models only win when fitted to 3D ground truth. They attribute this to an observation-noise bottleneck — monocular geometric ambiguity, not model expressiveness, is the limiting factor. Their multi-view error was 2.24 dm for the aerodynamic model versus 3.16 dm for a plain parabola, so the physics earns its keep only once the geometry is good.

Practical consequence: fit the 6-parameter model, report per-segment uncertainty, and resist adding drag and Magnus terms until there is a second view. Height is the weakly-constrained direction and should be reported with an interval, always.

### 2.9 Stage 8 — fusion, render, export (Tier C)

Pitch-space smoothing with the kinematics treatment from C5 (temporally consistent speed and acceleration), overlay rendering, and exports: per-frame tracks (parquet), ball 3D segments, calibration, events, and a QA report carrying abstention rate, calibration validity fraction, and per-link error probabilities.

**Optimise and report in pitch space, not pixel space** (D4). Equal pixel error on a distant player is a much larger metric error; a pixel-space loss silently prioritises the near touchline.

---

## 3. Compute, scale-out, and the live path

### 3.1 Per-frame budget

Engineering estimates for one modern GPU (L40S / RTX 4090 class, TensorRT FP16), 1080p25 input. To be replaced by measurements from the §5 harness.

| Stage | Invocation rate | Est. ms/call | ms per video-second |
|---|---|---|---|
| Decode (NVDEC) | 25 Hz | 0.8 | 20 |
| Shot / camera-type classification | 5 Hz | 2.0 | 10 |
| Player + referee detection | 25 Hz | 12.0 | 300 |
| Field keypoints + lines | 25 Hz | 8.0 | 200 |
| Ball detection | 25 Hz | 6.0 | 150 |
| Appearance embeddings (22 crops) | 5 Hz | 6.6 | 33 |
| Jersey OCR (selected keyframes) | 2 Hz | 4.0 | 8 |
| **Standard tier total** | | | **721** |
| Selective mask propagation | on ambiguity | — | 1 900–4 300 |

The mask layer is priced from A1's own figure — amortised 0.154–0.171 s/frame on an RTX 5090 — with the lower bound assuming the ~2× gain from object multiplexing (A9). **It costs 3–6× the entire rest of the pipeline.** That single fact drives three decisions: it is a separate quality tier; it runs in its own worker pool with its own queue so it cannot stall the main path; and the dispatch threshold on the assignment margin is a tuned cost/quality dial, not a constant.

### 3.2 The unit of parallelism is the camera shot

A broadcast shot is typically 3–15 s. Tracker state, motion models and calibration all reset at a cut anyway, so **shots are free shard boundaries** — no overlap frames, no seam artifacts, no stitching heuristics. Identity across shots is not Tier A's job; it is exactly what the Tier B tracklet graph exists to resolve.

This gives embarrassingly parallel scale-out with a clean correctness story: a 90-minute match is a few hundred independent Tier A jobs, dispatched to however many GPUs are available, followed by one cheap CPU job.

Estimated wall clock for a full 90-minute match, from the budget above assuming ideal sharding:

| GPUs | Standard tier | With mask layer |
|---|---|---|
| 1 | 65 min (1.4× real-time) | 4.0 h (0.4×) |
| 4 | 16 min (5.6×) | 60 min (1.5×) |
| 8 | 8 min (11×) | 30 min (3.0×) |
| 16 | 4 min (22×) | 15 min (6.1×) |

### 3.3 Engineering for scale-out

- Content-addressed cache keyed on `(shot_id, stage, config_hash, model_version)`. Changing the identity model re-runs Tier B only — seconds, not hours.
- Queue per stage with independent worker pools, so the expensive mask pool and the cheap OCR pool scale independently.
- Batch across shards, not only within a frame: the detector's efficient batch size is larger than the number of objects in one frame.
- Export TensorRT engines per model, per GPU architecture, built once in CI.
- Long-tail guard: any single shot that exceeds a time budget is returned with a degraded flag rather than blocking the match.

### 3.4 The live path, and the one stage that does not survive it

Tier A is already causal — nothing in it looks at future frames. Tier B is the problem, and there are two distinct cases:

- **Global tracklet association (A10's GTA) is unconditionally offline** — it needs completed tracklets. This is the single hardest constraint on a live version of the A1 stack, and the reason the quality tier is offline-first.
- **The HMM identity layer degrades gracefully.** Run it over a sliding window with a bounded lag (2–3 s) and it stays useful, because the constraints that make it strong (roster cardinality, motion feasibility) are available immediately.

So the live design is: Tier A unchanged, Tier B in sliding-window mode, and — the important part — an **output contract that permits revision**. Identity is emitted provisionally with a confidence, and the stream can carry a correction event when later evidence (a legible jersey number three seconds later) resolves an ambiguity. Any live consumer must handle an identity revision. This is strictly better than the alternative of committing to a guess, and it is what makes the offline and live products the same pipeline rather than two codebases.

Backward seed search (A1's walk to a clean frame) works live with the same bounded lookahead. Mask propagation at ~4 s of GPU per video-second needs roughly 4–6 GPUs per live stream, which is a cost decision, not an architectural one.

---

### 3.5 Deployment on Google Cloud Platform

The three tiers map onto three GCP services with no orchestration framework in between.

| Tier | GCP service | Why |
|---|---|---|
| A — per-shot GPU | **Cloud Batch**, one array job per match, one task per shot, GPU VMs (L4 for the standard tier, A100/H100 for the mask tier), Spot where allowed | Array jobs are the native shape of shot-level sharding: a few hundred independent tasks, each reading one shot from GCS and writing summaries back. Batch handles retries, quotas and Spot preemption; the content-addressed cache (§3.3) makes a preempted task safe to re-run. |
| B — match-global CPU | **Cloud Run job** (or a single CPU Batch task) triggered when all Tier A tasks are done | A few MB of tracklet features in, identities and 3D ball out. Seconds of runtime; no GPU. |
| C — fusion, render, demo | Render as a Cloud Run job; demo platform as a **Cloud Run service** that uploads to GCS and polls job state | The demo needs a public endpoint and no GPU of its own. |
| Storage | **GCS** bucket, layout `matches/<match_id>/shots/<shot_id>/<stage>/<config_hash>/` | Content-addressed by design; the same layout is the cache key. |
| Images | **Artifact Registry**, one image per tier; TensorRT engines built in Cloud Build per GPU family | Tier A and Tier B have different dependency stacks and very different sizes. |

Cost sketch at the standard tier, from the §3.1 budget: one full match is ~65 GPU-minutes on a 4090-class card; an L4 is slower per card, so expect roughly 1.5–2 L4-hours per match, or about 20 min wall-clock on 6 parallel L4 tasks. The mask tier multiplies GPU time by 3.5–7×, which is why it should run on Spot A100/H100 capacity and be the first thing turned off when a quota is tight. These figures become measurements after the first match runs; the Batch job log is the source.

Prerequisites on your side: a GCP project with a GPU quota in one region, a GCS bucket, and a service account whose key is registered in this workspace (Customize → Credentials) so the pipeline driver can submit Batch jobs and read/write the bucket. Region choice should follow GPU availability, not proximity.

Two consequences for the code: every stage is a CLI that takes `--input-uri --output-uri --config` and nothing else, so the same binary runs on a laptop, in a Batch task or in a Cloud Run job; and the driver that fans out shots and waits for completion is the only GCP-specific component, isolated behind an interface so a Slurm or local-multiprocess backend can replace it.

## 4. The difficulties

Each difficulty: what actually breaks, why the obvious answer is insufficient, what we do, and what risk remains.

### 4.1 Occlusion

**What breaks.** Players occlude each other constantly — corners, walls, celebrations, any duel. Three distinct failures, usually conflated:

1. *Missed detection.* The detector returns nothing. A tracker that treats a miss as evidence of absence deletes the track.
2. *Corrupted measurement.* The box shrinks and shifts toward the visible part, so the positional cost misleads the association step (A4's "positional cost confusion") — worse than a miss, because it is confidently wrong.
3. *Corrupted appearance.* The crop contains two players; its embedding is a blend that will match neither later.

**Why the obvious answer fails.** Raising detector recall trades occlusion misses for false positives in crowds. A bigger Kalman covariance keeps the track alive but widens the gate so much that it grabs the wrong player.

**What we do.**

- Correct the positional cost from an estimated occlusion state (A4) instead of trusting a shrunken box.
- Treat occlusion as part of the sensor model, not a patch on association: each object's probability of detection accounts for the presence of all other objects (A8). A miss that is *predicted by occlusion geometry* should cost the track nothing; only an unexplained miss is evidence of absence. This is the formalism the identity belief in §2.7 is built on.
- Down-weight occluded crops before they enter the appearance bank, by per-part visibility (C3) — never blend an occluded crop into a track's appearance prototype at full weight.
- Dispatch mask propagation exactly where the assignment margin says the tracker is unsure (A1), and let it override only on confident contradiction.
- Amodal completion (F3) is available as an *association cue only*. It is generative and therefore hallucinates; a completed mask may be used where only temporal consistency is needed, and must never be used as a measurement for position, pose or joint angles.

**Measured by.** AssA and IDF1 on a held-out broadcast set; identity switches per minute; and a stratified breakdown by occlusion level, because aggregate HOTA hides exactly the cases we are trying to fix.

**Residual risk.** Dense multi-player pileups with total mutual occlusion over 2+ seconds — a corner-kick scramble. Expect the system to abstain there rather than to be right; the abstention is the mitigation.

### 4.2 Identity

**What breaks.** Same kit, same build, same motion, 22 of them, for 90 minutes with frequent exits from frame. The reference for how hard this is: identity loss in comparable livestock settings occurs 10–20 times per minute (F2). And a single identity error is not a local error — it corrupts every event attributed to that trajectory until recovery (C1).

**Why the obvious answer fails.** Appearance ReID alone cannot separate teammates; that is the defining property of the problem (D1 was designed around it). Jersey OCR alone is far too sparse — a number is legible in a small minority of frames, and never during the occlusion where you most need it. Per-frame greedy assignment throws away the constraints that actually determine the answer.

**What we do.** The §2.7 stack, in priority order of what does the work:

1. **Global, not greedy.** Identity is resolved on the tracklet graph over the whole match (or a sliding window, live), not frame by frame. The 2026 gains in this area are in the association layer, not the embedding (E3).
2. **Roster cardinality as a hard constraint.** ≤11 per team, one goalkeeper each, monotone substitutions, permanent dismissals. Strongly informative and almost free.
3. **Sparse uncertain evidence handled as such** (C4's HMM), not as a hard label whenever OCR happens to fire.
4. **Keyframe selection** for OCR rather than restoration of poor crops (F4).
5. **Abstain under the CSIS loss** (C1) instead of guessing.
6. **Revisable identity** in the output contract (§3.4), so a late-arriving legible number can correct earlier frames rather than being discarded for consistency.

**Measured by.** CSIS on LTPI (C1) — the only open benchmark for full-match player identity, with data, code and weights released — plus the abstention rate and the fraction of match-time each roster slot is confidently held.

**Residual risk.** Goalkeeper/referee confusion at sequence start before clusters stabilise (E3's named failure mode) — mitigated by detecting them as classes rather than clustering them. Players whose number is never legible in the whole match: they get a stable anonymous identity, which supports tactical analysis but not player attribution. Say so in the output rather than inventing a name.

### 4.3 Camera and broadcast structure

**What breaks.** A moving, zooming, cutting monocular camera with graphics overlaid. Cuts to close-ups and replays. Score bugs and lower-thirds covering players. Letterboxing. Variable frame rate and interlacing depending on broadcaster.

**Why the obvious answer fails.** Always-on camera-motion compensation adds noise during static holds; per-frame independent homography estimation jitters because it has no temporal prior; and a pipeline with no notion of shots will happily track a graphic.

**What we do.** Stage 0 classification and cut detection (§2.1); conditional motion compensation (A3); pan/tilt/zoom-space temporal smoothing (§2.3); a per-broadcast profile auto-detected at ingest (letterbox geometry, static graphic regions to mask, frame rate, field of view statistics); replay segments processed but tagged, and excluded from match-time aggregation by default.

**Residual risk.** Extreme zoom close-ups where the pitch model is unconstrained — calibration is marked invalid rather than reported badly. Handheld and cable-cam shots behave differently from the tripod model; these are detected and treated as uncalibrated.

### 4.4 The ball

**What breaks.** Small (8–15 px), fast (up to ~30 m/s), blurred, frequently occluded by legs and bodies, out of frame on long balls, and visually confusable with pitch line markings, white socks, distant heads and background objects. Then monocular 3D: depth along the camera ray is weakly observable.

**Why the obvious answer fails.** Deblurring hurts more than it helps at light blur (F4). Per-frame detection has no way to distinguish the ball from a white sock; only motion continuity does. And a higher-capacity aerodynamic model does not fix monocular 3D, because the bottleneck is observation noise, not model expressiveness (B2).

**What we do.** §2.6 for detection (signed motion priors, temporal aggregation, visibility-weighted loss, explicit state machine, blur as signal); §2.8 for 3D (contact-segmented fitted-gravity physics with a reprojection objective, uncertainty always reported, no aerodynamic terms until a second view exists).

**Measured by.** Detection accuracy stratified by visibility — specifically accuracy on fully occluded frames, which is the number that separates a usable ball track from a decorative one (B1) — and, for 3D, reprojection error plus height interval width on the segments where triangulated ground truth exists (B2's released soccer datasets).

**Residual risk.** Aerial duels where the ball is hidden at the apex; long passes leaving frame entirely. Both are handled as state (`occluded`, `out-of-frame`) with interpolation flagged as interpolation.

### 4.5 Motion that defeats the motion model

**What breaks.** Constant-velocity Kalman assumptions are wrong for football: sprint, stop, turn, jump. This is one of the two properties SportsMOT was deliberately built around (D1).

**What we do.** Expanded-IoU association designed for irregular sports motion (A10); pitch-space physical limits instead of pixel-space gates (§2.4). Two upgrade paths if the association layer proves to be the bottleneck: a learned association module replacing hand-crafted Kalman+IoU+cosine heuristics (A5, +3.2% HOTA on SportsMOT), or a transformer motion predictor replacing the linear filter (F2's TransTrack-OC-SORT). One caveat before wiring in pose cues: keypoints *degraded* performance on SportsMOT because distant broadcast viewpoints make pose estimation noisy (A5). Test on our footage before adopting.

### 4.6 Small, distant players

**What breaks.** A far-side player is 20–30 px tall; the detector's recall there is much lower and its localisation error, projected to the pitch, is much larger.

**What we do.** Resolution first (D4), boundary-aware adaptive tiling (E2), pitch-space loss and pitch-space evaluation so the far side is not silently deprioritised, and calibration-derived size priors to gate implausible detections.

**Residual risk.** In wide shots, positional precision for the far-side full-back will be visibly worse than for near players. Report per-region accuracy rather than a single match number; a single number here is misleading by construction.

### 4.7 Generalisation to any broadcast

**What breaks.** Kit colours and patterns, third kits, bibs; night matches under floodlights with hard shadows and colour casts; rain, snow, low winter sun; different leagues' camera conventions; broadcaster graphics; codec artifacts; 25 vs 30 vs 50 fps; 720p vs 1080p vs 4K.

**Why the obvious answer fails.** Per-video threshold tuning is the standard research shortcut and it is exactly what makes a pipeline undeployable — your reference calls it out as the weakness of tracking-by-detection (A2). Anything requiring manual per-match calibration or a kit vocabulary violates §1.3.

**What we do.**

- **Training-free association components wherever possible** (A1, A3, A4 are all training-free, all plug-and-play) — the reason they were chosen over marginally stronger trained alternatives.
- **Relative team clustering**, never a fixed kit vocabulary (§2.5).
- **Per-match auto-calibration of the few things that must adapt**: pitch dimensions, broadcast profile, cluster assignment. All from unsupervised signals within the match, no human in the loop.
- **A hard-case regression set** as a standing gate: night games, rain, snow, low sun, unusual kits, heavy graphics, 4K and 720p sources, at least one match per broadcaster convention. Every model change runs against it.
- **Multi-sport / multi-source training where evidence supports it** — multi-sport training gave +19.2% tennis mAP and +14.6% badminton over single-sport specialists for ball tracking (B8). Worth testing for football-only ball detection across competitions and camera styles.

**Residual risk.** A truly novel broadcast style (an unusual camera position, a heavily stylised feed) will degrade. Detect it rather than absorb it: monitor calibration validity fraction and detection-count distributions per match, and flag out-of-distribution inputs instead of returning confident nonsense.

### 4.8 Throughput and latency

Covered quantitatively in §3. The architectural answers: selective dispatch so the expensive model runs on the hard fraction only; shot-level sharding for linear scale-out; separate worker pools per stage; asymmetric invocation rates (25 Hz detection, 5 Hz appearance, 2 Hz OCR); and a quality tier that is explicitly priced so the accuracy/cost decision is made with numbers rather than by default.

**Residual risk.** The 0.72 s/video-second base estimate is an estimate. If measured cost is materially higher, the dials in priority order are: detector resolution, appearance embedding rate, and mask dispatch threshold — in that order, because that is the order of cost per unit of accuracy.

### 4.9 Knowing which outputs to trust

**What breaks.** A pipeline that reports positions without confidence pushes the entire burden of trust onto the consumer, who cannot see the ambiguity. Downstream, one identity switch changes a player's distance covered, sprint count and pass attribution for the rest of the match.

**What we do.** Per-link error probabilities, following the most principled treatment of this problem in your reference — which comes from cell tracking (F1). Their insight transfers exactly: the Hungarian cost matrix already contains the alternatives, and the gap to the second-best assignment is the raw material for a calibrated error probability. A1 already exploits that gap as a dispatch signal, in uncalibrated form; calibrating it turns it into a reportable confidence. Two payoffs, both from F1: manual review is limited to the rare low-confidence links, and fully automated analysis becomes possible by retaining only high-confidence segments.

Concretely: every track link, every identity assignment and every calibration frame carries a probability; the QA report summarises them; and downstream analytics gate on them rather than assuming the input is clean.

### 4.10 Annotation and ground truth

**What breaks.** Bounding boxes are ambiguous in exactly the crowded, occluded scenes that matter — two annotators will disagree on the extent of a half-hidden player, and that disagreement becomes label noise precisely where the model needs signal.

**What we do.** Annotate with **masks rather than boxes** where we annotate at all (F2's recommendation, for exactly this reason: masks have unambiguous ground truth under occlusion). Prefer public data first: TeamTrack and PFF FC's synchronised 2022 World Cup event-and-tracking release for all 64 matches at 29.97 Hz are ungated (D5) and cover most of what SoccerNet-GSR would give without the licence friction; LTPI (C1) for identity; B2's released soccer datasets for ball 3D. Our own annotation budget then goes to the hard-case set (§4.7), not to re-annotating what already exists.

### 4.11 Licensing and data rights

A design constraint, not a footnote, since the pipeline is intended to ship. See §7.

---

## 5. Evaluation protocol

### 5.1 Fix the protocol before comparing anything

Your reference makes this point and it is worth restating as a rule: SportsMOT numbers are not all measured in the same setting, and offline methods that post-process finished tracklets (86.8 HOTA with global tracklet association) are not comparable to genuinely online trackers (72.2 HOTA, no external data). Before any internal benchmark we fix: detector, training data, online vs offline, and whether tracklet-level post-processing is permitted. Every reported number carries that configuration.

### 5.2 Metrics per stage

| Stage | Primary metric | Secondary |
|---|---|---|
| Detection | recall of on-pitch persons @ IoU 0.5, stratified by pitch region | false positives outside the pitch polygon |
| Tracking | HOTA, AssA, IDF1 | identity switches per minute; stratified by occlusion level |
| Identity | CSIS (C1) | abstention rate; confident roster coverage over match time |
| Calibration | median pitch-space reprojection error (m) | fraction of frames with valid calibration |
| Ball 2D | accuracy stratified by visibility, incl. fully occluded frames | state-machine confusion matrix |
| Ball 3D | reprojection error; height interval width | fraction of segments rejected |
| End-to-end | pitch-space position RMSE per region | throughput (× real-time), latency (live mode) |

### 5.3 Acceptance targets for milestone 1

Targets, not measurements — set from the published reference points and to be revised once the harness runs on your footage.

- Tracking on a held-out broadcast set: HOTA ≥ 78, IDF1 ≥ 82 (reference: Deep-EIoU 77.2 HOTA online on SportsMOT, D1).
- Calibration: valid on ≥ 95% of main-camera frames; median pitch-space reprojection error ≤ 0.30 m within the visible region.
- Ball: detected or correctly stated as occluded/out-of-frame on ≥ 95% of main-camera frames.
- Identity: confident, correct identity on ≥ 90% of visible player-frames, with abstention (not error) on the remainder.
- Throughput: ≥ 5× real-time per match on 4 GPUs, standard tier.

### 5.4 Regression discipline

Every model or config change runs the hard-case set (§4.7) and reports the full metric table. A change that improves aggregate HOTA while degrading occluded-case AssA is a regression, and the stratified tables are what make that visible.

---

## 6. Data plan

| Purpose | Source | Note |
|---|---|---|
| Development, ungated | TeamTrack; PFF FC 2022 World Cup (D5) | 279,900+ frames / 4.37M boxes; synchronised tracking + events for 64 matches |
| Identity benchmark | LTPI (C1) | full 101-minute match, track-level ground-truth identities, code + weights released |
| Ball 3D ground truth | B2's released soccer datasets | triangulated 3D, segment-level annotations |
| Tracking benchmark | SportsMOT (D1) | reference points for every tracker in §2.4 |
| Actions / roles | FOOTPASS (D2) | 54 matches, 102,992 frame-level annotations — but CC BY-NC and video under SoccerNet NDA (§7) |
| Hard-case regression | our own | night, rain, snow, low sun, unusual kits, 4K/720p, multiple broadcasters |

---

## 7. Licence register

Chosen per your direction — best model now, obligations tracked for cleanup before commercial release.

| Component | Licence | Risk if shipped | Replacement path |
|---|---|---|---|
| RF-DETR (detection) | Apache-2.0 | none | — |
| YOLO26 (alt. detection) | AGPL-3.0 | requires paid commercial licence | RF-DETR is the default for this reason |
| SAM 3 / 3.1 (mask propagation) | SAM licence, not Apache | must be reviewed before shipping | quality tier is optional by design; open amodal alternatives exist (F3, Apache-2.0) |
| DINOv3 (features) | Meta agreement for commercial use | needs agreement | KPR/OSNet cover ReID; DINOv3 is not the bottleneck for team classification (E3) |
| Deep-EIoU, GTA, CAMELTrack, KPR, TrackLab | open repos, check each | low | — |
| FOOTPASS annotations | CC BY-NC 4.0 | non-commercial only | fine for R&D and publication; not for the product |
| SoccerNet video | NDA, redistribution prohibited | cannot ship or redistribute | benchmark internally only; ship on PFF FC / own footage |

The one structural decision this register forces: **the quality tier must be removable.** If the SAM licence review fails, the standard tier still ships and the accuracy delta is known from the A/B.

---

## 8. Roadmap

| Milestone | Content | Exit criterion |
|---|---|---|
| M0 | Evaluation harness, metric implementations, hard-case set, protocol fixed (§5.1) | every metric in §5.2 computable on one match with one command |
| M1 | Standard tier, offline: stages 0–5 + naive identity | §5.3 targets met except identity |
| M2 | Identity layer: tracklet graph, HMM, cardinality, CSIS abstention | CSIS on LTPI competitive with C1 baselines |
| M3 | Ball 3D: contact segmentation + physics fit with uncertainty | reprojection error and interval widths reported on B2 data |
| M4 | Quality tier: selective mask propagation, A/B against M2 | measured HOTA/IDF1 delta and its measured cost |
| M5 | Live mode: sliding-window Tier B, revisable identity, streaming API | end-to-end latency ≤ 1.5 s at a stated GPU count |

The demo platform (video in, overlay out) is built on the M1 outputs and upgraded as later milestones land — it is the acceptance test the whole pipeline is judged by, so it should exist as early as M1.

---

## 9. Open questions

Resolved:

1. ~~Is full 22-player state a requirement?~~ **No.** Visible players are the priority; off-frame estimation is out of scope. Re-entry identity is the hard requirement (§1.2).
2. ~~Quality or latency for milestone 1?~~ **Quality.** Mask tier on by default; Tier B window long; live deferred to M5.

3. ~~Jersey-number attribution?~~ **Yes** — stable identity plus jersey number when legible. Step 9 is in scope from M2.
4. ~~Annotation on own footage?~~ **No** — quality on your footage is judged visually from the demo overlay; quantitative numbers come from public benchmarks (LTPI, SportsMOT, B2). Consequence: the overlay must expose what the numbers would have — identity confidence, abstentions, re-entry links, calibration validity — otherwise visual review cannot see the failures that matter (§2.9).
5. ~~GCP GPU?~~ **L4 (g2)** for the standard tier. The mask tier will be slow on L4; profile it there first, request A100 quota only if the measured cost justifies it.
6. ~~Demo hosting?~~ **Cloud Run service** on the same GCP project (§3.5).

Still open:

7. **What competitions and broadcasters must work on day one?** This defines the hard-case set, which is the only thing that will actually keep §4.7 honest. Can be answered by the first videos you download.
8. **GCP project details:** project id, region with L4 quota, bucket name, service-account credential registered in the workspace, Spot acceptable or not.

---

### Reference codes

Entry codes (A1–A10, B1–B8, C1–C5, D1–D5, E1–E3, F1–F4) refer to *Computer Vision for Team Sports — Curated paper reference*, Quantum Sports Analytics, compiled 3 September 2026, supplied as `CV_Team_Sports_Paper_Reference.pdf`.
