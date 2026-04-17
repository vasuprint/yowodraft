# YOWOFormer Evaluation

This directory contains evaluation scripts for YOWOFormer on different datasets.

## AVA Dataset Evaluation

We use the **official AVA evaluation metrics** from Google Research for reliable and reproducible results suitable for academic papers.

### Usage

```bash
# Basic evaluation with AVA standard settings
python evaluate_ava_official.py --checkpoint weights/ava_test/best.pth

# With explicit parameters (using standard values)
python evaluate_ava_official.py \
    --checkpoint weights/ava_improved_v2/best.pth \
    --batch_size 32 \
    --conf_thres 0.01 \
    --nms_thres 0.5 \
    --save_json
```

### Parameters

- `--checkpoint`: Path to trained model checkpoint (required)
- `--batch_size`: Batch size for evaluation (default: 32)
- `--conf_thres`: Confidence threshold for detections (default: **0.01** - AVA standard)
- `--nms_thres`: NMS IoU threshold (default: **0.5** - standard for duplicate removal)
- `--data_root`: Path to AVA dataset (default: data/AVA_Dataset)
- `--num_workers`: Number of dataloader workers (default: 6)
- `--save_json`: Save results to JSON file

### Standard Settings for Paper Comparison

This evaluator uses the **official AVA evaluation protocol** with standard settings used in major papers:

| Parameter | Standard Value | Used By |
|-----------|---------------|---------|
| **conf_thres** | 0.01 | SlowFast, YOWOv2, ACAR-Net, AIA |
| **nms_thres** | 0.5 | YOWOv2, YOWOv3 |
| **IoU threshold** | 0.5 | All AVA papers |
| **Max detections** | 100/frame | AVA protocol |

These values ensure **fair comparison** with other published methods.

### Output

The evaluation script will:
1. Generate detections in CSV format
2. Run official AVA evaluation using PascalDetectionEvaluator
3. Report mAP@0.5 and per-category AP
4. Save results to JSON (if --save_json is used)

### Multi-Label Support

This evaluator properly handles AVA's multi-label nature:
- Each person can perform multiple actions simultaneously
- The model outputs are processed with sigmoid activation
- All action predictions above the threshold are saved (not just argmax)

### Files Generated

- `ava_detections.csv`: Raw detections in AVA format
- `ava_evaluation_results.json`: Evaluation metrics (if --save_json)

## UCF101-24 Dataset Evaluation

For UCF101-24, use the existing evaluation script:

```bash
python evaluate_videomae.py --checkpoint weights/ucf_test/best.pth --dataset ucf
```

## Citation

If you use this evaluation code in your research, please cite:

```bibtex
@inproceedings{gu2018ava,
  title={Ava: A video dataset of spatio-temporally localized atomic visual actions},
  author={Gu, Chunhui and Sun, Chen and Ross, David A and Vondrick, Carl and others},
  booktitle={CVPR},
  year={2018}
}
```

## Credits

The AVA evaluation code is adapted from:
- Google Research AVA Dataset (Apache License 2.0)
- TensorFlow Object Detection API
- YOWOv3 implementation