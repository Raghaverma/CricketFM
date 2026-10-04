# Annotation guide

Two label sets, both private (`data/`, gitignored). Labels are the ceiling of every number in
this repo: a model can't be measured more precisely than two careful people agree, so part of
each set is labeled twice, blind, and that agreement is reported beside the model results.

## Pose (PoseFM)

### Workflow

```bash
python -m cricketfm.data.inventory                 # data/videos.csv
python -m cricketfm.data.sampling motion           # motion traces + data/pose/sheets/<video_id>.jpg (CPU)
# fill the per-video fields below in data/videos.csv, one row per video, from its contact sheet
python -m cricketfm.data.sampling select           # 800 frames: data/pose/images/*.jpg + data/pose/to_annotate.json
# annotate in a COCO-keypoint tool (CVAT: create a task from data/pose/images with the COCO-17
# skeleton and a `phase` attribute, or import to_annotate.json), export "COCO Keypoints 1.0", then:
python -m cricketfm.data.pose_gt merge <export.json>   # restores the metadata tools drop -> data/pose/annotations.json
```

### Per-video fields (`data/videos.csv`)

Filled once per video, before sampling: the sample is stratified by them (`docs/PROTOCOL.md` §3.3).
`select` refuses to run while any eligible video lacks them.

| Column | Values | Notes |
|---|---|---|
| `view` | `front_on`, `side_on`, `rear`, `oblique` | where the camera is, relative to the person labeled (table below) |
| `content` | `cricket`, `test`, `screen`, `empty`, `other` | `test`: app testing, no cricket; `screen`: filming a screen |
| `rotation` | `none`, `rotated`, `mixed` | a sideways or upside-down scene |
| `batting_hand` | `right`, `left`, `mixed`, `unknown` | batting videos; `mixed`: several batters |
| `bowling_arm` | `right`, `left`, `mixed`, `unknown` | bowling videos; `mixed`: several bowlers |
| `lighting` | `daylight`, `indoor`, `floodlit`, `poor` | `poor`: dim, dusk, or strongly backlit |
| `usable` | `no` to exclude | not cricket, rotated, or the person is never in frame |

| view | camera, relative to the person labeled |
|---|---|
| `front_on` | in front of them, looking down the pitch at them: at the bowler's end for a batter, at the batter's end for a bowler |
| `side_on` | square of the pitch, level with them |
| `rear` | behind them, looking the way they face |
| `oblique` | anything in between, roughly 30-60 degrees off those |

Frames are named `<video_id>_<frame index>.jpg`. The index is the frame's position in a
sequential decode of the source (`cricketfm/data/video.py`), never a timestamp: most of this
footage is variable-frame-rate.

### Who

- **Batting videos:** the striker only. **Bowling videos:** the bowler only.
- Not the non-striker, wicketkeeper, umpire, coaches, or people in neighbouring nets.
- If the person is truncated by the frame edge, label what is in frame.
- If the person is unrecognisable (motion blur over the whole body, mostly out of frame), skip
  the image. Skipped images are counted.

### Box

Tight around the body, including helmet, pads and gloves, **excluding the bat**.

### Keypoints: COCO-17, COCO's anatomical conventions

| Flag | Meaning | Coordinates |
|---|---|---|
| `2` visible | the joint is in view | where it is |
| `1` occluded | hidden, but its position can be estimated with confidence | the estimate |
| `0` not labeled | out of frame, or position can't be estimated | 0, 0 |

Cricket-specific rules:

- **Equipment worn on the body is not occlusion.**
  - A knee under a pad's knee roll is visible (`2`) at the joint centre behind the roll.
  - The same applies to an ankle in a boot or a wrist inside a glove.
  - Objects in front of the body are occlusion: the **bat** in front of a wrist or a knee, or
    the bowler's own arm crossing the face.
- **Wrists:** at the glove cuff, where the forearm meets the hand, not the centre of the glove.
  - On the handle the hands overlap. The far wrist is usually `1`, placed just above or below
    the near one.
- **Helmet:**
  - Eyes seen through the grille are `2`.
  - Ears under the helmet's ear guards are `1` if their position is clear from the helmet's
    orientation, else `0`.
- **Hips:** COCO places hips at the top of the thigh, at the hip joint, not at the waistband.
  This is COCO's least consistent joint, so follow its examples rather than intuition.
- **Self-occlusion:** in side-on stance the far shoulder, hip, knee and ankle are behind the
  body. Mark them `1` with an estimate when the pose makes the position clear; `0` otherwise.

### Phase tag

One attribute per person, `phase`, judged from the frame (look at the neighbouring frames in the
video when unsure). It is a reporting stratum; the sampler's motion bin only approximates it.

| Role | `phase` |
|---|---|
| striker | `stance` (set, waiting for the ball), `stroke` (backlift to contact), `follow_through`, `between` (not facing a delivery) |
| bowler | `run_up`, `delivery` (back-foot landing to release), `follow_through`, `between` (walking back, waiting) |

A second attribute, `hand` (`right` or `left`), is the person's batting hand or bowling arm. A
video can show several people, so this comes from the person, not the video.

`pose_gt merge` prints how many people carry a phase tag; check it on the first export.

### Agreement

- 80 images (10%), flagged `double_annotate` in `to_annotate.json`, go to a second annotator who
  doesn't see the first annotator's labels. They are spread across strata in proportion.
- Their OKS against the first annotator, per keypoint, is the noise floor. A model "better"
  than it on some keypoint is matching one annotator's habits, not the body.

## Shots (ShotFM)

One record per delivery the striker faces, in `data/shots/clips.jsonl` (fields:
`cricketfm/data/clips.py`).

- **label:** the stroke the batter **attempted**, whether or not bat met ball, from the fine
  labels in `configs/shots.yaml`.
  - A cover drive played and missed is `cover_drive`.
  - Off side and leg side are the batter's, so they swap with handedness.
  - Use `unclear` when it can't be judged. Those clips are counted, then excluded.
- **contact_frame:** the first frame in which the ball touches bat, glove, pad or body. For a
  leave or a miss, it's the frame in which the ball passes the batter. The single-frame and
  three-frame conditions are cut around it.
- **start_frame / end_frame:** 1.5 s before contact to 1.0 s after (`fps_avg` from
  `data/videos.csv`, rounded to frames), clamped to the video. That is 2.5 s, enough for the
  backlift and the follow-through.
- **batting_hand:** right or left.
- **view:** where the camera is, relative to the batter (the pose table above, with the batter
  as the person): `front_on` is at the bowler's end, `rear` behind the batter or keeper.
- **flags:** the conditions that hold for the clip, possibly none (`[]`). Tagged on every clip,
  before any model output exists: they are the base rates of the failure analysis.

  | flag | when |
  |---|---|
  | `occlusion` | something between the camera and the batter (another player, a net pole) hides the bat or a limb around contact |
  | `poor_framing` | the batter is cut off by the frame edge, or under about a tenth of the frame height |
  | `motion_blur` | the bat or arms are smeared at contact |
  | `bat_not_visible` | the bat can't be seen at the contact frame |

- **borderline:** `true` when the label was a close call between two strokes (e.g. a cover drive
  or a square drive). Optional, default false.

- **Agreement:** 20% of clips are labeled by a second person, blind. Cohen's kappa is reported
  at both levels, fine labels and v1 classes. Disagreements are resolved by discussion into the
  final label, and the pre-discussion labels are kept for the kappa.
