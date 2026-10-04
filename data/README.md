# data/

Everything here except this file is gitignored: the footage, frames and labels are private.
Committed results refer to footage by AutoClipping job id and content fingerprint only.

```
data/
├── coco -> ../../TensorRT/datasets/coco   COCO val2017 + annotations/person_keypoints_val2017.json (public)
├── videos.csv                             one row per AutoClipping job video (data/inventory.py);
│                                          fill in view, batting_hand / bowling_arm, lighting, usable
├── pose/
│   ├── motion/<video_id>.npy              per-frame motion proxy (data/sampling.py motion)
│   ├── sheets/<video_id>.jpg              contact sheet per video, for filling videos.csv
│   ├── images/<video_id>_<frame>.jpg      frames picked for annotation (data/sampling.py select)
│   ├── to_annotate.json                   COCO images + metadata, no labels: import into the tool
│   └── annotations.json                   labeled set: tool export + metadata (data/pose_gt.py merge)
└── shots/
    └── clips.jsonl                        one record per delivery (data/clips.py, docs/ANNOTATION.md)
```

The source videos stay where AutoClipping keeps them (`../AutoClipping/data/jobs/<job_id>/source.mp4`);
nothing here copies or modifies them.
