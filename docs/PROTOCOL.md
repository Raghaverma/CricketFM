# CricketFM benchmark protocol

**Benchmarking foundation models and specialist vision systems for cricket understanding.**

**Status: v1.0** (2026-10-04). The decisions left open in v0.1 are recorded in §8, made before
any cricket label or result existed. The protocol is frozen by tagging the commit `protocol-v1`.
After that, a change gets a new version and a line in §11 saying what changed and why. A change
made after seeing test results is marked as such.

## 1. Questions

There are three questions. Everything else in this document serves one of them.

| | Question | Experiment | Answered by |
|---|---|---|---|
| Q1 | **Foundation vs specialist:** does Sapiens give better cricket pose estimates than ViTPose++? | A, PoseFM | Paired AP and PCK differences on the cricket set (§4.1); the same pair on COCO val2017, to see whether the ranking changes out of domain; cost per image (§4.3) |
| Q2 | **Generalization:** can a general-purpose VLM recognize cricket shots without cricket training? | B, ShotFM | Qwen3.5-4B zero-shot under three input conditions, against chance (§4.2) |
| Q3 | **Adaptation:** how much does LoRA improve shot recognition, from how much data, at what compute cost? | B, ShotFM | Macro-F1 against clips per class, from zero-shot up (§4.2); the specialist on the same subsets; GPU-hours and inference cost |

v0.1's other two questions are folded into these. Cost (v0.1's RQ4) is part of each answer:
- pose inference for Q1;
- VLM inference per input condition for Q2;
- LoRA training and inference for Q3.

Whether generality pays (RQ5) is read from Q1's COCO-vs-cricket ranking and from Q3's VLM and
specialist curves at equal data.

**What counts as an answer:**
- **Q1:** "Sapiens is better" or "worse" on a metric only when the 95% interval of the paired
  difference excludes 0. Otherwise the answer is "no difference detected", with the interval
  as the bound on it.
  - Q1 needs Sapiens' COCO-17 head (§5). The Goliath head's result is baseline evidence, not
    an answer (§10).
- **Q2:** zero-shot macro-F1 against two chance baselines (§4.2): a permutation null that keeps
  the model's own answer frequencies, and the majority class.
- **Q3:**
  - the paired difference LoRA − zero-shot at each training budget;
  - the budget at which the curve flattens;
  - the specialist's curve on the same subsets.

**Expectations, written before any cricket result.** They are here so a result can be read as
confirming or refuting them; either outcome is a finding.

- **Pose:** Sapiens (COCO-17 head) stands better against ViTPose++ on cricket than on COCO.
  - Its margin grows, or its deficit shrinks, out of domain, mostly on occluded joints (bat,
    pads) and wrists.
  - Its cost is 16x the input pixels.
- **Zero-shot VLM:** well below a pose-based specialist. An independent May 2026 benchmark
  (VideoNet, arXiv 2605.02834) found open 4-8B VLMs close to chance on its 41 cricket actions.
- **Temporal context:** frames3 and clip beat frame. VideoNet also found the centre frame alone
  never beat the clip.

## 2. Experiments

```
                                        CricketFM
              ┌─────────────────────────────┴──────────────────────────────┐
   Experiment A: PoseFM (Q1)                         Experiment B: ShotFM (Q2, Q3), pilot
   ViTPose++  vs  Sapiens (COCO-17 head)             L1  VLM zero-shot, one frame
   same frames, same GT boxes, COCO-17               L2  VLM zero-shot + temporal context
   800 frames: 400 striker, 400 bowler               L3  VLM + LoRA, data-efficiency curve
                                                     L4  ViTPose++ pose -> temporal classifier
                                                     L5  VLM + pose overlay
              └───────────── accuracy · latency · memory · energy ─────────────┘
```

- **PoseFM doesn't wait for ShotFM.** It has its footage on this box now. ShotFM is a pilot
  (§3.4).
- **ShotFM's levels run in order.** L5 runs only after L1-L4 are reported, so the baselines are
  fixed before pose is added to the VLM's input.
- **All ShotFM levels share** the same clips, folds, per-class training subsets and 8 classes.
- **The VLM is Qwen3.5-4B** at every level (§5).

## 3. Data

### 3.1 What exists on this box

Source: AutoClipping's job videos, `results/data_inventory.json` (`python -m cricketfm.data.inventory`).

| | Unique videos | Players | Footage | VFR | Mostly |
|---|---|---|---|---|---|
| Batting (all `STANCE_ANALYSIS`) | 62 | 10 | 19.0 min (median video 13.5 s) | 35 | 1280x720, 60 fps |
| Bowling | 56 | 13 | 97.0 min | 20 | 1080p / 720p |

- **Duplicates and failures:** 33 of 157 job videos are re-uploads of footage already present
  (same content fingerprint), and 6 are unreadable.
- **Players:** 7 people appear in both modes.
- **Players are concentrated:** one batter accounts for 20 of the 62 batting videos.
- **No labels of any kind exist:** no keypoints, no shot types, no camera views. The job
  metadata's `angle` is "bowler" on every job and is not a camera angle.
- **The batting videos are stance-analysis uploads.** Nobody has counted how many shots they
  contain. AutoClipping's shot detector has never run here because the bat model is missing.
- **Reviewed by eye** (2026-10-04, recorded in `data/videos.csv`):
  - **Content:** of the 118 unique readable videos, 74 show cricket. 30 are app tests in an
    office or a room, 5 film a screen, 8 show something else (a road, a desk), and 1 has no
    bowler.
  - **Usable:** 71 videos are upright cricket footage: 30 batting (10.8 min, 3 accounts) and
    41 bowling (93.6 min, 7 accounts).
  - **Burned-in overlays:** many batting uploads are recordings of the app's live view, with its
    skeleton and coaching text boxes drawn into the pixels.
    - That is 66% of all batting frames and 0.7% of bowling frames.
    - Detection (`cricketfm/data/overlay.py`) agreed with the eye on 50 of 50 frames, and
      flagged 0 of 9,465 frames in four clean videos.
    - Overlay frames carry another model's pose, so PoseFM never samples them. A ShotFM model
      can't be shown them either: a VLM would see a pose overlay, the L5 condition.
  - **`player_id` is the uploading account, not the person.** Some accounts hold many people:
    batters taking turns in a nets session, or different bowlers across one account's clips.
    Caps and groups "per player" are therefore per account.
  - **What's left under the §3.3 caps:** 85 striker frames (17 videos, 3 accounts) and 384
    bowler frames (41 videos, 7 accounts).
- **The footage set is fixed** by `data/videos.csv` as it stands when the pose sample is drawn.
  Jobs uploaded to AutoClipping after that go into an expansion, not into v1.

### 3.2 Rules for all data

- **Frames are decoded indices.** A frame is its position in a sequential decode of the
  source, never a timestamp. VFR phone video drifts by over a second (AutoClipping measured
  that on 21 of 54 batting sources).
- **Splits are group-disjoint.** No player and no video (by content fingerprint) appears on
  two sides of a split. Records sharing either are joined by union-find
  (`cricketfm/data/splits.py`), so re-uploads can't leak.
- **Privacy:**
  - Footage, frames, labels and per-frame outputs stay in `data/` and `results/raw/`
    (gitignored). Committed results name footage by job id and fingerprint, never by path.
  - Identity is the job metadata's pseudonymous `player_id`. No display names or URLs are
    copied.
  - A frame appears in the report only with the player's consent. Otherwise it is shown as a
    pose skeleton or described.
- **Per-video facts are filled in once** in `data/videos.csv`:
  - camera view, relative to the person labeled;
  - handedness: `batting_hand` for batting videos, `bowling_arm` for bowling videos;
  - lighting: `daylight`, `indoor`, `floodlit` or `poor`;
  - `usable`.

  The contact sheets from `sampling motion` (`data/pose/sheets/`) show enough to judge all
  four.

### 3.3 PoseFM: the pose set

- **Size: 800 frames,** 400 striker (from batting videos) and 400 bowler (from bowling videos).
  720 frames are labeled once and 80 twice.
- **Built for hard, varied poses,** not for the footage's natural mix, which is mostly a batter
  standing side-on in stance. A set of clean side-on stances would say little about cricket.
- **Strata:**

  | Variable | Source | How it is controlled |
  |---|---|---|
  | Player | job metadata | Sampling: equal share per player within a cell, at most 120 frames per player per cell; the largest per-player total is reported |
  | Camera view | `videos.csv` | Sampling: equal share per cell (view × handedness × lighting) |
  | Handedness | `videos.csv`: `batting_hand`, `bowling_arm` | Sampling, as above |
  | Lighting | `videos.csv` | Sampling, as above |
  | Action phase | Before labeling, a motion proxy; after it, the annotator's phase tag | Sampling: each video's frames alternate between its most-moving quarter and the rest. Reported by phase tag |
  | Batting stance | The striker's `stance` phase; stance style is a property of the batter | Through phase and player |
  | Distance from camera | Person box height, from the labels | Reported only |
  | Occlusion | Visibility flags, from the labels | Reported only |

- **No model output picks frames.** Distance and occlusion need a person box and joints, which
  means either the labels or a model, so they are reporting strata only. Selecting frames on a
  pose model's output (say, low ViTPose++ confidence) would build a test set adversarial to one
  side of Q1.
- **The motion proxy is model-free:**
  - For each decoded frame, it is the mean absolute difference between the frames 0.1 s before
    and after it (greyscale, 160 px wide).
  - Within a video, frames alternate between the top quartile of its motion and the rest,
    starting with either at random.
  - The high-motion share is a little under half: 46% of both roles in a dry run on the real
    traces. High-motion frames come in short bursts, and the 1 s gap limits how many one burst
    can give.
  - **It may be weak on batting footage.** In three sampled stance videos, the 90th
    percentile of motion was 1.8-2.7x the median. The proxy measures the whole frame, so camera
    shake and people in neighbouring nets count as well as the batter's swing.
  - So it's only a sampling lever. Results are stratified by the phase tag, and the
    motion-bin × phase-tag table is reported as a check on the proxy.
- **Allocation** (`cricketfm/data/sampling.py`) is equal allocation capped by availability, down
  the tree role → cell → player → video.
  - Each node splits its quota equally among its children.
  - A child with fewer frames than its share gives all it has, and the rest is split among the
    others.
  - Rare cells (a rear view, a left-hander, poor light) therefore contribute everything they
    have.
- **Within a video:** at most 16 frames, at least 1 s apart, drawn with a seed from the video
  id.
- **Overlay frames are never drawn,** nor any frame within 0.5 s of one (§3.1).
- **Handedness is tagged per person** at labeling. `videos.csv` records `mixed` for a session
  with several people and `unknown` when a video doesn't show it.
- **The sample is most of what exists.** A dry run on the real traces finds 528 striker and
  459 bowler frames available under these caps, so 800 uses about 80% of them.
  - Stratification can't enlarge a rare cell beyond its footage. Its main effects are the
    motion split and the player balance.
  - Raising the per-video cap barely helps: at 32 per video, the bowler side only grows to 538,
    because the player cap binds first.
- **What the overall AP describes:** this enriched set, not the footage's natural mix.
  - Paired model differences are valid on it, and every stratum is reported (§4.1).
  - Each frame records its cell and motion bin, so a reweighted estimate stays possible.
- **Skipped frames** (no labelable person) are counted. If skips exceed 5%, each skipped frame
  is replaced by the next frame in the same video's draw order (`reserve` in
  `to_annotate.json`).
- **Labels:** COCO-17 keypoints with visibility flags, the person's box, and a phase tag,
  following `docs/ANNOTATION.md`.
- **Agreement:** 80 frames (10%) are labeled again by a second annotator, blind.
  - They are a systematic sample over the selection sorted by stratum, so they spread across
    strata in proportion.
  - The per-keypoint OKS between the two annotators is the noise floor.
- **Everything is test data.** No model is trained in PoseFM.
  - Any analysis that fits a parameter (e.g. the convention-offset correction in §4.1) fits on
    a player-disjoint half and reports on the other half.
  - COCO val2017 (public) is the second domain for Q1's ranking comparison.

### 3.4 ShotFM: the shot set (a pilot)

- **ShotFM is a pilot** (decision 1 in §8). It uses all eligible batting footage (19 min, 10
  batters), which should give on the order of 100 clips.
  - Every ShotFM result is labeled as a pilot on a small set. It estimates effect sizes and
    finds failure modes.
  - It does not support claims about cricket-shot recognition in general.
  - If the pilot shows informative failure modes, the set is expanded (target ≥ 30 batters)
    under a new protocol version.
- **Records:** `data/shots/clips.jsonl` (`cricketfm/data/clips.py`).
- **Labels:** annotated at the fine level (14 strokes plus `unclear`, in `configs/shots.yaml`)
  and evaluated at the v1 level: `cover_drive`, `straight_drive`, `pull`, `cut`, `flick`,
  `sweep`, `defensive`, `other`.
  - `hook` maps to `other`, not to `pull` (decision 4 in §8).
  - The class set is a versioned mapping over the fine labels, so it can change without
    re-annotating.
  - 20% of clips are double-labeled, and Cohen's kappa is reported.
- **Clip conditions** are tagged on every clip at labeling, before any model has seen it:
  - `flags`: occlusion, poor framing, motion blur, bat not visible at contact;
  - `borderline`: the label was a close call.

  With the camera view, they give the failure analysis its base rates (§4.4).
- **Evaluation is grouped K-fold over players** (K = 5), not one fixed split.
  - With 10 batters, a train/val/test split would test on two or three people.
  - With K-fold, every clip is predicted once, by a model that never saw its player. Metrics
    pool the out-of-fold predictions.
  - Results are reported per fold as well as pooled. With about two players per fold the folds
    will differ a lot, and that spread is part of the result.
  - The fixed train/val/test split (`splits.assign_splits`) applies once an expanded set has
    enough batters.
- **Input conditions:**
  - `frame` (L1): the contact frame.
  - `frames3` (L2): contact - 200 ms, contact, contact + 200 ms. It is the primary condition
    for L3-L5. The plan's ±2 frames is ±33 ms at 60 fps, which gives three near-identical
    images.
  - `clip` (L2): the 2.5 s clip, subsampled to the highest of 8, 4 and 2 fps at which LoRA
    training on Qwen3.5-4B still fits beside production on the L4.
    - Training is the tighter limit, and zero-shot and LoRA must see the same input.
    - A memory test on unlabeled footage fixes the rate before any labeled clip is run.
  - `frames3_pose` (L5): the `frames3` images with the striker's COCO-17 skeleton drawn on
    them.
    - The skeleton comes from L4's detector and ViTPose++-L, not from the labels, so L5 is a
      system that could be deployed.
    - The drawing is fixed now: COCO-17 edges, 2 px lines, left and right sides in two colours.

## 4. Metrics

Every interval is a 95% percentile bootstrap that resamples **clusters**, not frames
(`cricketfm/evaluation/stats.py`): videos in PoseFM, players in ShotFM, images on COCO.
Comparisons between two systems are paired, with both scored on the same resampled clusters.
ShotFM's intervals resample 10 players. With that few clusters a percentile interval is
approximate, usually too narrow, so the cluster count is printed beside every interval.

### 4.1 Pose (`cricketfm/evaluation/pose_metrics.py`)

- **Primary:**
  - **AP** over OKS, COCO's definition (AP, AP50, AP75). With one annotated person per frame
    and ground-truth boxes it has a closed form that equals pycocotools' COCOeval (tested).
  - **PCK@0.05**, normalized by box size (max of width and height).
- **Secondary:**
  - PCK by body region: head, torso, arms, legs.
  - PCK on visible vs occluded joints.
  - PCK per keypoint.
  - torso-normalized PCK@0.2, mean OKS.
  - **Joint-angle error** in degrees at elbows, knees and shoulders: what a coaching product
    reads off a pose.
- **Keypoint conventions:**
  - Mapping by name fixes the format (`cricketfm/skeleton.py`), but not where each training set
    puts a joint.
  - Each model's mean signed offset per keypoint (in box units) is reported beside its error. A
    model whose error is mostly a constant offset has a convention gap, not a recognition gap.
  - A bias-corrected PCK is computed as a secondary analysis: the offset is fitted on one half
    of the players and the correction applied to the other half.
- **Strata** (`evaluation/pose_eval.py`):
  - role;
  - camera view, handedness, lighting;
  - motion bin and phase tag;
  - box height in pixels (the distance stratum);
  - share of occluded joints.

  A stratum with fewer than 20 people is listed but not scored.

### 4.2 Shots (`cricketfm/evaluation/shot_metrics.py`)

- **Primary:** macro-F1 over the v1 classes. Class balance here is whatever the footage holds.
- **Secondary:** accuracy, balanced accuracy, per-class F1 and the confusion matrix.
- **Per fold:** every metric is also reported per test fold, with the mean, SD and range across
  folds (`per_fold`).
- **Chance baselines** (Q2):
  - **Majority class:** each fold's most frequent training class, predicted for every test clip
    (`majority_predictions`).
  - **Permutation null:** the system's own predictions shuffled 10,000 times
    (`permutation_test`). The shuffle keeps how often the system names each class and cuts any
    link to the clip's stroke. The p-value is one-sided.
    - **Q2's test shuffles within each player.** A zero-shot model can't learn a batter's habits
      from this data. But its answers can still follow the camera setup, which goes with the
      batter, and so with that batter's mix of strokes. Shuffling within a player keeps that
      and removes only the stroke.
    - The shuffle across all clips is reported beside it.
- **Invalid responses:**
  - A VLM answer is read strictly: valid JSON naming a class (`cricketfm/models/vlm/parsing.py`).
  - Anything else is invalid and scored as wrong, never dropped. The invalid rate is reported.
  - A lenient reading, which also accepts prose naming exactly one class, is reported as a
    diagnostic only.
- **Confidence:** the VLM's self-reported confidence is auxiliary. Its calibration (ECE) is
  measured, not assumed.
- **Prompt:**
  - Generated from the taxonomy (`cricketfm/models/vlm/prompts.py`) and used as generated. Every
    result records its `prompt_id`.
  - No prompt search is done on the pilot set, which is too small to both tune and test on.
  - The same prompt is used before and after LoRA.
- **Temporal ablation** (L1-L2, and L3 at the full budget):

  | Input | Macro-F1 | Latency per clip | Input tokens |
  |---|---|---|---|
  | `frame` | | | |
  | `frames3` | | | |
  | `clip` | | | |

  If `frames3` or `clip` beats `frame` (a paired difference), the model uses motion, not only
  a static batting pose.
- **Data-efficiency curve** (Q3):
  - **Budgets are in clips per class.**
    - Full study: 25, 50, 100, 200.
    - Pilot: powers of two from 2 up to the largest k that at least half the classes reach in
      every training fold (`splits.budget_ladder`), plus the whole training fold.
    - The pilot ladder is fixed from label counts alone, before any model is trained.
    - Zero-shot is the k = 0 point.
  - A class with fewer clips than k contributes all it has, and the actual count per class is
    recorded.
  - **Subsets** are nested and drawn inside each training fold (`splits.per_class_subsets`).
    LoRA and the specialist train on the same subsets.
  - **Runs:** the `frames3` condition, with 3 training seeds per point. The seed spread is
    reported separately from the player interval.
  - **One LoRA recipe** is fixed before the first run and recorded in `configs/`. Any tuning
    uses inner grouped folds of the training players only, never the test fold.
  - **The plot:** macro-F1 against clips per class (log x), zero-shot at 0, the specialist on
    the same axes, and GPU-hours per point.

### 4.3 Cost (`cricketfm/benchmark/pose_latency.py`)

| Measure | Definition |
|---|---|
| Model latency | Forward pass on a preprocessed batch already on the GPU; mean / p50 / p95 / p99 ms |
| Throughput | Images/s at batch 1, 2, 4, 8, 16 |
| End to end | Frame + box → keypoints, split into preprocess / forward / decode (ms per frame); video decode and detection excluded, as they're identical for every pose model |
| Memory | Weights, input batch, and activation peak above both (MB); the CUDA context is excluded |
| Energy | Mean NVML board power × latency / batch (J per image); SM clock recorded |
| VLM inference | Per clip and input condition: latency (prefill + generation), input tokens, peak memory; batch 1 |
| Training | GPU-hours and peak memory per LoRA or classifier fit, per budget |

### 4.4 Failure analysis

- **Confusion matrices** for every system and condition, with invalid answers as their own
  column.
- **Failure causes are defined now,** before any error is seen, so they can't be fitted to the
  errors. They come in two kinds.
- **Clip conditions**, tagged on every clip at labeling (§3.4):
  - occlusion;
  - poor framing;
  - motion blur;
  - bat not visible;
  - unusual camera view (`rear` or `oblique`).

  For each condition, every system's error rate is reported on clips with it and on clips
  without it, with player intervals. A share of errors means little without its base rate.
- **Error types**, assigned to each wrong prediction afterwards:

  | Type | Rule |
  |---|---|
  | Similar stroke geometry | True and predicted classes share a bat plane (vertical: drives, flick, defensive; horizontal: pull, cut, sweep; `configs/shots.yaml`) |
  | Insufficient temporal context | Wrong on `frame`, right on `frames3` or `clip`, same model |
  | Taxonomy ambiguity | The clip is marked `borderline`, or its two labelers disagreed |
  | Hallucination | The answer contradicts plainly visible evidence (e.g. `sweep` with the batter upright), as judged by a reviewer who doesn't know which system answered |
  | Invalid output | No valid answer (§4.2) |
  | Unexplained | None of the above |

  An error can have several types.

## 5. Models

| Experiment | Model | Source | Native input | Role |
|---|---|---|---|---|
| A | ViTPose++ S/B/L/H, COCO expert (`dataset_index` 0) | HF `usyd-community/vitpose-plus-*` | 256x192 | specialist |
| A | Sapiens 0.3B/0.6B/1B, COCO-17 head | `noahcao/sapiens-pose-coco` (gated) | 1024x768 | foundation, same labels as ViTPose++ |
| A | Sapiens 0.3B/0.6B/1B, Goliath-308 head → COCO-17 | `facebook/sapiens-pose-*-torchscript` | 1024x768 | foundation, own labels; baseline evidence |
| B | Qwen3.5-4B | HF, Apache-2.0 | images and video | L1-L3, L5: zero-shot, then LoRA |
| B | detector → ViTPose++-L striker pose → temporal classifier | this repo | pose sequence | L4, specialist |

- **Primary pose pair:** Sapiens-COCO vs ViTPose++ at matched backbone size.
  - The pairs are ViTPose++-L with Sapiens-0.3B (~0.3B), and ViTPose++-H with Sapiens-0.6B
    (~0.6B).
  - Both learned COCO's labels, so what differs is the model itself (human-centric
    pretraining, 16x the input pixels), not the label convention.
  - The Goliath head shows what the foundation model gives without a COCO fine-tune, convention
    gap included.
- **If the COCO-17 checkpoints can't be used** (licence or access, §8), PoseFM reports the
  Goliath head only, as baseline evidence, and Q1 stays open.
- **The VLM is Qwen3.5-4B** at every level.
  - A 4B model is a more meaningful foundation-model baseline than a model shrunk to make the
    experiment easier.
  - A smaller model is added only if the 4B can't run LoRA beside production on the L4. The
    memory test in step 8 (§9) settles it, and its result is reported.
- **Same input, same output:**
  - PoseFM accuracy uses ground-truth boxes, so the detector is out of the comparison.
  - Every model gets the same frames and boxes, through its own reference crop and decode, at
    its own input size.
  - Sapiens follows its evaluation configs (UDP affine crop at w/h 0.75), not its lite demo,
    which stretches people on 16:9 frames.
  - Neither model uses a flip test. The published Sapiens COCO numbers use one, and also use
    other detections than ViTPose's, so they don't compare with ViTPose's published numbers.
- **The specialist** normalizes pose before classifying: centred on the mid-hip, scaled by
  torso length, left-handers mirrored (`skeleton.mirror`).
- **Deferred:** Sapiens2 (April 2026, transformers-native, 308 keypoints only). Its licence
  prohibits "biometric processing"; it is added once that is cleared.

## 6. Timing protocol

- **Hardware:** the shared NVIDIA L4 (24 GB, 72 W power cap), driver 595.91, CUDA 13.0,
  PyTorch 2.14, transformers 5.17. Every result file records its environment
  (`cricketfm/benchmark/env.py`).
- **Same optimization level for every model:** PyTorch eager, FP16, no TensorRT and no
  `torch.compile` in the primary comparison. FP16 is checked against FP32 per model; for
  Sapiens it moves keypoints 0.02 px median, 0.35 px max. An optimized deployment comparison
  (the TensorRT repo's methods) is a separate, later row.
- **Resolution:** each model at its native input size. The size is part of the model; a fixed
  shared size would cripple one of them.
- **Per cell (model × batch size):**
  - 20 warm-up calls discarded, then 100 timed calls, each bracketed by device syncs.
  - 2 rounds, interleaved across models, because the L4's clocks drift under its power cap.
  - `cudnn.benchmark` off. p50 and p95 reported, never the first call.
- **VLM latency** follows the same guard and warm-up rules, per input condition, at batch 1.
- **Contention guard** (`cricketfm/benchmark/gpu_guard.py`, ported from the TensorRT repo):
  - A cell starts only when no AutoClipping production job holds a GPU slot and the box has
    been quiet for 15 s.
  - A cell that overlapped anything is discarded and re-run.
  - Models release their memory while a production job runs.
  - Untimed GPU work (accuracy runs, LoRA training) also waits for production and yields to
    it.

## 7. The final table

ShotFM rows are marked as pilot results.

| System | Pose AP / shot macro-F1 | Latency | VRAM | Energy | Cricket training data |
|---|---|---|---|---|---|
| ViTPose++-L (COCO) | | | | | none |
| Sapiens-0.3B, COCO-17 head (Goliath head as baseline) | | | | | none |
| L1-L2 VLM zero-shot (`frame` / `frames3` / `clip`) | | | | | none |
| L3 VLM + LoRA | | | | | k clips per class (curve) |
| L4 ViTPose++ + temporal classifier | | | | | the same subsets |
| L5 VLM + pose overlay (zero-shot / LoRA) | | | | | none / whole training fold |

## 8. Decisions (2026-10-04)

1. **ShotFM is a pilot,** run now on all eligible footage.
   - Kept: 5-fold player CV, per-fold results, and intervals as far as 10 players allow.
   - The set is called small wherever a result appears.
   - Recording ≥ 30 batters or licensing other footage is the expansion path if the pilot
     finds failure modes worth following. Neither gates the pilot.
2. **Sapiens:** the primary pair needs the COCO-17 checkpoints, and the Goliath result is
   baseline evidence (§10).
   - Pending action (step 1 in §9): confirm that a khel.ai research benchmark is within
     CC-BY-NC-4.0, then accept the gated repo's terms.
   - The design doesn't depend on the outcome. §5 says what is reported without access.
3. **Pose annotation: 800 frames,** 720 labeled once and 80 twice, stratified as in §3.3.
   - The annotators are named in the results.
   - The second annotator is a different person, blind to the first annotator's labels.
4. **Taxonomy v1: `hook` stays in `other`.**
   - Hook and pull look alike but are distinct strokes.
   - Merging hook into `pull` would make one evaluated class mean two strokes that labeling
     keeps apart.
   - In `other`, hook sits with the other strokes the 8 classes don't cover.
5. **VLM: Qwen3.5-4B,** zero-shot and then LoRA. A 2B model is added only if the 4B can't run
   beside production.

## 9. Execution order

PoseFM comes first: it has its data, and the small shot set doesn't hold it up.

| # | Step | State |
|---|---|---|
| | ViTPose++ and Sapiens (Goliath) adapters, COCO gates, timing; protocol; ShotFM fold design | done (§10) |
| 1 | Sapiens COCO-17: licence decision, then gated access (`huggingface-cli login`, accept terms) | open |
| 2 | Fill `data/videos.csv` (view, handedness, lighting) from the contact sheets (`sampling motion`) | |
| 3 | Draw the pose sample (`sampling select`) | |
| 4 | Annotate 800 frames; 80 a second time, blind | |
| 5 | PoseFM: ViTPose++ vs Sapiens COCO-17 (and the Goliath head) on cricket and COCO | |
| 6 | Label the ShotFM pilot clips | |
| 7 | L1-L2: Qwen3.5-4B zero-shot on `frame`, `frames3` and `clip` | |
| 8 | LoRA pipeline; memory test (clip fps, 4B fits beside production) | |
| 9 | L3: LoRA data-efficiency curve | |
| 10 | L4: ViTPose++ temporal classifier on the same subsets | |
| 11 | Final table (§7) | |
| 12 | Failure analysis (§4.4) | |
| 13 | L5: pose overlay, zero-shot and LoRA | |
| 14 | Report and figures | |

## 10. Phase 0 checks (done)

| Check | Result |
|---|---|
| ViTPose++-L on COCO val2017 (6,352 people, GT boxes, no flip) | AP **0.8087**, the TensorRT repo's Stage 10 reproduced (0.809); every keypoint PCK@0.1 ≥ 0.919 (`results/coco_check/vitpose-plus-l.json`) |
| Matched AP vs pycocotools on single-person images | identical to 1e-9 (`tests/test_pose_metrics.py`) |
| Sapiens Goliath-308 → COCO-17 mapping | names read by executing Sapiens' own dataset config: wrists at 62/41, no duplicates |
| Sapiens codec | frame → crop → heatmap → decode round trip within 0.25 px (`tests/`) |
| Sapiens on a batch-1 trace | batch 4 runs; batched vs single max abs heatmap diff 0.001 (fp32), 0.007 (fp16) |
| Sapiens-0.3B, Goliath head, on COCO val2017 | passes the mapping gate (lowest keypoint PCK@0.1 0.824), AP **0.667** vs ViTPose++-L's 0.809 (`results/coco_check/sapiens-0.3b-goliath.json`) |
| Cricket pose eval plumbing | `pose_eval` run end to end on an 80-image stand-in set: strata, video-cluster intervals, paired differences |
| Cost, Sapiens-0.3B vs ViTPose++-L (PyTorch FP16) | 4.4x (batch 1) to 21x (batch 16) fewer images/s, 21x the energy per image; ViTPose++-L within 8% of the TensorRT repo's Stage 4; 0 of 20 guarded cells re-run (`results/pose_latency/l4_vitpose-l_vs_sapiens-0.3b.json`) |

**On this setup, the specialist beats the general model on accuracy, throughput, latency and
energy at once.** Only on weights does Sapiens-0.3B come out smaller (629 vs 829 MB in FP16).

**That is baseline evidence, not the answer to Q1.** It compares a foundation model's head that
never saw COCO labels with a specialist trained on them:
- **Not an offset.** No keypoint's mean offset exceeds 0.6% of box size, while every
  keypoint's mean error is 1.5-1.9× ViTPose++'s.
- **Most likely a labelling-style gap.** A head that never saw COCO's annotation style, scored
  on COCO's own labels, loses about 0.14 AP to a model trained on them.
  - These data can't separate that from Goliath's training images being further from COCO's.
  - The cricket labels follow COCO conventions (`docs/ANNOTATION.md`), so the same home
    advantage would carry over to cricket.
- **Hence the primary pair** is Sapiens' COCO-17 head against ViTPose++ (§5). The Goliath row
  is reported as "the foundation model without a COCO fine-tune".

## 11. Change log

- v0.1 (2026-10-04): first draft.
- v1.0 (2026-10-04): the decisions in §8, made before any cricket label or result existed.
  - The questions are reduced to Q1-Q3. The tracks are renamed PoseFM and ShotFM.
  - ShotFM runs as a pilot with levels L1-L5. `frames3` is its primary condition, and the
    `clip` frame rate follows a fixed rule.
  - The pose set is 800 frames, stratified, with motion-enriched sampling; it no longer caps
    each video at 8 uniformly drawn frames.
  - Taxonomy v1 is confirmed (`hook` → `other`). Qwen3.5-4B is the only VLM.
  - Added: per-fold reporting, chance baselines, the data-efficiency curve in clips per class,
    the temporal ablation with latency, the failure analysis with base rates, the pose-overlay
    condition, and the footage snapshot.
