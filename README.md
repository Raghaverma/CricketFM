# CricketFM

**Benchmarking foundation models and specialist vision systems for cricket understanding.**

| | Question | Experiment |
|---|---|---|
| Q1 | Foundation vs specialist: does Sapiens give better cricket pose estimates than ViTPose++? | A, PoseFM |
| Q2 | Generalization: can a general-purpose VLM recognize cricket shots without cricket training? | B, ShotFM |
| Q3 | Adaptation: how much does LoRA improve shot recognition, from how much data, at what compute cost? | B, ShotFM |

- **PoseFM:** Meta's Sapiens, a human-centric foundation model, against ViTPose++, a specialist
  pose model. Both run on the same 800 cricket frames with the same person boxes, and are scored
  in COCO-17.
- **ShotFM, a pilot:** Qwen3.5-4B classifies the batter's stroke into 8 classes at five levels:
  - zero-shot on one frame;
  - zero-shot with temporal context;
  - LoRA over a data-efficiency curve;
  - a ViTPose++ temporal classifier as the specialist;
  - the VLM with a pose overlay.

The benchmark is defined in [docs/PROTOCOL.md](docs/PROTOCOL.md): questions, data, metrics,
timing protocol, decisions and execution order. v1.0 records every decision, made before any
cricket label or result existed. It is frozen by tagging the commit `protocol-v1`.

## Status

Steps follow `docs/PROTOCOL.md` §9.

| Step | State |
|---|---|
| Protocol | v1.0, all decisions made |
| ViTPose++, Sapiens (Goliath head) | Adapters in place; COCO gates passed; latency measured |
| 1. Sapiens COCO-17 head | Waits on a licence decision (CC-BY-NC-4.0) and gated access |
| 2-3. Pose sample | Stratified sampler ready; needs view, handedness and lighting per video in `data/videos.csv` |
| 4-5. Pose benchmark | Tooling ready end to end; needs 800 labeled frames ([docs/ANNOTATION.md](docs/ANNOTATION.md)) |
| 6. Shot pilot | Schema (with clip conditions), taxonomy, group-disjoint folds and per-class budgets ready; needs labels |
| 7-9. VLM zero-shot, LoRA | Prompt, strict parser, metrics, per-fold and chance baselines ready |
| 10-14. Specialist, final table, failure analysis, pose overlay, report | Not started |

## Results so far

### Every pose adapter must pass COCO first

A wrong crop, a transposed decode or a scrambled keypoint mapping still produces plausible
keypoints. `cricketfm.evaluation.coco_check` scores each adapter on COCO val2017: 6,352
people, ground-truth boxes, no flip test, pycocotools AP. It fails a model that misses a
reference result, or that scores any keypoint below 0.5 PCK@0.1.

| Model (input) | COCO AP | Mean OKS | Occluded-joint PCK@0.05 | Lowest keypoint PCK@0.1 | Wall time |
|---|---|---|---|---|---|
| ViTPose++-L, COCO expert (256x192) | **0.8087** | 0.883 | 0.626 | 0.919 (left hip) | 93 s |
| Sapiens-0.3B, Goliath-308 head mapped to COCO-17 (1024x768) | 0.6674 | 0.797 | 0.434 | 0.824 | 645 s |

- **ViTPose++-L reproduces** the TensorRT repo's Stage 10 (0.809), which used the same
  processing.
- **Sapiens' mapping is correct:** no keypoint collapses.
- **The 0.14 AP gap is not a constant offset.** Every keypoint's mean offset is under 0.6% of
  box size, while its mean error is 1.5-1.9× ViTPose++'s. That is consistent with a head that
  never saw COCO's labelling style being scored on COCO's own labels.
- **This is baseline evidence, not the answer to Q1.** Here the specialist beats the
  foundation model on accuracy, throughput, latency and energy at once. But it is a head that
  never saw COCO labels against one trained on them.
- **Hence the primary pair** is Sapiens' COCO-17 head against ViTPose++
  (`docs/PROTOCOL.md` §5, §10). The COCO-17 weights are gated.

### What the foundation model costs on this L4

Measurement setup:
- PyTorch eager, FP16, each model at its native input, run on distinct COCO crops.
- Every cell ran under the contention guard; none of the 20 needed a re-run.
- Two interleaved rounds, with a round-to-round spread of at most 6%.
- Results: `results/pose_latency/l4_vitpose-l_vs_sapiens-0.3b.json`, plus `.csv` with the
  plan's record fields.

| Batch | ViTPose++-L img/s | Sapiens-0.3B img/s | ViTPose++ advantage |
|---|---|---|---|
| 1 | 51.3 | 11.7 | 4.4x |
| 4 | 198.3 | 10.7 | 18.5x |
| 16 | 222.3 | 10.4 | 21.4x |

| | ViTPose++-L | Sapiens-0.3B |
|---|---|---|
| Energy per image, batch 16 | 0.32 J | 6.88 J |
| Activation peak, batch 16 | 216 MB | 2,920 MB |
| Weights, FP16 | 829 MB (all six MoE experts) | 629 MB |
| End to end, one person per frame | 27.6 ms (preprocess 7.4, forward 20.0, decode 1.7) | 90.8 ms (preprocess 1.2, forward 85.0, decode 4.7) |

- **Sapiens doesn't batch.** At 16x the pixels (3,072 tokens per crop against 192), a single
  crop already holds the L4 at its 72 W power cap (about 1.07 GHz at batch 1), so its
  throughput is flat from batch 1. ViTPose++-L runs at full clock until batch 4.
- **The timing matches the TensorRT repo:** ViTPose++-L is within 8% of that repo's Stage 4
  PyTorch column at every batch size.

### What the footage on this box supports

`python -m cricketfm.data.inventory` reads AutoClipping's job videos without decoding them.
Results are in `results/data_inventory.json`.

| | Unique videos | Players | Footage | Variable frame rate |
|---|---|---|---|---|
| Batting | 62 | 10 | 19.0 min | 35 |
| Bowling | 56 | 13 | 97.0 min | 20 |

- **Duplicates:** 33 of the 157 job videos are re-uploads of footage already present, and
  splits treat them as one video.
- **No labels yet:** none exist of any kind.
- **PoseFM is feasible:** its 800 frames come from 118 videos.
- **ShotFM runs as a pilot:** 19 minutes of batting from 10 batters should give on the order of
  100 shots, not the plan's 500-2,000. Its results are reported as a pilot on a small set
  (decision 1 in the protocol).

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
ln -s ../../TensorRT/datasets/coco data/coco    # COCO val2017 + person_keypoints_val2017.json
huggingface-cli login                           # only for the gated Sapiens COCO-17 checkpoints
```

The pinned versions match the TensorRT repo's venv (torch 2.14, CUDA 13.0, transformers 5.17),
so numbers in the two repos are measured on the same stack.

## Commands

```bash
pytest                                                              # CPU tests
python -m cricketfm.evaluation.coco_check --model vitpose-plus-l      # adapter gate (~2 min)
python -m cricketfm.benchmark.pose_latency --models vitpose-plus-l sapiens-0.3b-goliath
python -m cricketfm.data.inventory                                   # footage -> data/videos.csv
python -m cricketfm.data.sampling motion                             # motion traces + contact sheets (CPU)
python -m cricketfm.data.sampling select                             # 800 stratified frames to annotate
python -m cricketfm.data.pose_gt merge <tool export.json>            # labels -> data/pose/annotations.json
python -m cricketfm.evaluation.pose_eval --models vitpose-plus-l sapiens-0.3b-coco sapiens-0.3b-goliath
```

Models are named in `configs/pose_models.yaml`. The shot taxonomy, which is also the source of
the VLM prompt, is in `configs/shots.yaml`.

## Layout

```
cricketfm/
  skeleton.py         COCO-17, regions, joint angles, name-based mapping from native skeletons
  taxonomy.py         fine shot labels -> versioned class sets
  data/               inventory, stratified frame sampling, decoded-index video access, pose labels,
                      shot clips, group-disjoint splits and per-class training budgets
  models/pose/        ViTPose++ (HF) and Sapiens (TorchScript) behind one preprocess/forward/decode interface
  models/vlm/         prompt from the taxonomy, strict answer parser
  evaluation/         pose and shot metrics, cluster bootstrap, timing, COCO gate, cricket pose eval
  benchmark/          contention guard and guarded runner (from the TensorRT repo), latency sweep, environment
configs/              models, taxonomy, timing protocol
docs/                 PROTOCOL.md, ANNOTATION.md
results/              committed aggregates (JSON); results/raw/ holds per-frame outputs and stays local
data/                 private: footage index, frames, labels (see data/README.md)
```

## Conventions

- **Footage stays private.** Footage, frames and labels are never committed. Result files name
  footage by job id and content fingerprint, and only aggregates are committed.
- **Frames are decoded indices,** never timestamps: most phone footage here is
  variable-frame-rate.
- **Production comes first on the GPU.** It is shared with AutoClipping's production service,
  so every GPU job yields to it (`benchmark/gpu_guard.py`). Timed cells are re-run if anything
  overlapped them.
- **Intervals resample clips, videos or players,** never single frames.
