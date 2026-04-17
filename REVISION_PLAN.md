# YOWOFormer Revision Plan (IEEE Access)

## Phase 1: Setup RunPod
- [ ] Extract frames AVA dataset (step3_extract_frames.sh)
- [ ] เตรียม UCF101-24 dataset
- [ ] Push code ขึ้น GitHub
- [ ] git clone บน RunPod
- [ ] ทดสอบ train script ทำงานได้

---

## Phase 2: Experiments (ต้อง run ใหม่)

### R1-Q4: SQA ไม่มี self-attention (2 runs)
- แก้โค้ด: เพิ่ม `use_self_attention=False` ใน CrossAttentionAdapter
- เพิ่ม `--no_self_attention` flag ใน train script
- [ ] Run 1: UCF101-24 — YOLOv11n + MAE-B + SQA(no self-attn) + Simple (เทียบ 91.43%)
- [ ] Run 2: AVA v2.2 — YOLOv11n + MAE-B + SQA(no self-attn) + Simple (เทียบ 22.42%)

```bash
# UCF
python train_videomae_stable.py \
  --yolo_version n --videomae_method cross --fusion_module Simple \
  --dataset ucf --epochs 20 --batch_size 12 --scheduler cosine \
  --freeze_videomae --unfreeze_epoch 6 --lr 0.0001 \
  --gradient_accumulation 4 --gradient_clip 0.5 --weight_decay 0.0001 \
  --no_self_attention --save_dir weights/ucf_sqa_no_selfattn

# AVA
python train_videomae_stable.py \
  --yolo_version n --videomae_method cross --fusion_module Simple \
  --dataset ava --epochs 20 --batch_size 12 --scheduler cosine \
  --freeze_videomae --unfreeze_epoch 6 --lr 0.0001 \
  --gradient_accumulation 4 --gradient_clip 0.5 --weight_decay 0.0001 \
  --no_self_attention --save_dir weights/ava_sqa_no_selfattn
```

### R1-Q5: Parallel BSTF (2 runs)
- แก้โค้ด: สร้าง ParallelBSTF variant ใน fusion module
- [ ] Run 3: UCF101-24 — YOLOv11n + MAE-B + SQA + Parallel BSTF
- [ ] Run 4: AVA v2.2 — YOLOv11n + MAE-B + SQA + Parallel BSTF

### R1-Q6: TAL top-k sensitivity (2 runs)
- แก้โค้ด: เปลี่ยน top_k ใน TAL config
- [ ] Run 5: AVA v2.2 — top_k=5 (เทียบ default k=10)
- [ ] Run 6: AVA v2.2 — top_k=15

---

## Phase 3: แก้ Paper (ไม่ต้อง run experiment)

### Reviewer 1
- [ ] R1-Q1: เพิ่มเหตุผลเลือก YOLOv11 ใน Section III-B (C2PSA, benchmark, Table 4)
- [ ] R1-Q2: อธิบาย 1568 tokens = (T/2) x (H/16) x (W/16) = 8 x 14 x 14
- [ ] R1-Q3: อธิบาย 49 queries = 7x7 grid design choice + flexibility discussion
- [ ] R1-Q7: เพิ่มข้อความ FPS = forward pass only ใน Section IV-B
- [ ] R1-Q8: อ้าง prior work ที่ใช้ split 1 เป็น standard (ไม่ต้อง run split อื่น)
- [ ] R1-Q9: เพิ่ม 2 references (YOLOMF, YOLODF) ใน Section I + III-B
- [ ] R1-Q10: อธิบาย 2D backbone ใหญ่ขึ้นแย่ลง (overfitting + temporal bottleneck)
- [ ] R1-Q11: ขยาย future work ใน conclusion

### Reviewer 2
- [ ] R2-Q1: ยืนยัน evaluation protocol (same split/IoU/metric)
- [ ] R2-Q2: เพิ่ม inference settings ใน Section IV-A + code availability
- [ ] R2-Q3: ชี้ไปที่ Table 1-3 ที่มี Params/GFLOPs/FPS อยู่แล้ว
- [ ] R2-Q4: เพิ่ม SQA justification (เทียบ DETR-style, simple reshape)
- [ ] R2-Q5: ชี้ไปที่ Fig. 7-10 ที่มีอยู่แล้ว
- [ ] R2-Q6: เพิ่ม code availability statement + GitHub link
- [ ] R2-Q7: เพิ่ม Limitations section

---

## Phase 4: Response to Reviewers
- [ ] Draft response letter (ภาษาอังกฤษ)
- [ ] ตรวจทานทั้ง paper + response
- [ ] Submit revised manuscript

---

## Training Config (ใช้เหมือนกันทุก run)
- Optimizer: AdamW, lr=1e-4, wd=1e-4
- Scheduler: Cosine
- Epochs: 20
- Freeze VideoMAE epoch 1-5, unfreeze last 6 layers epoch 6 (0.1x LR)
- EMA decay=0.999, FP16, grad accum 4, grad clip 0.5
- Batch size: 12 (effective 48)
- Resolution: 224x224, clip=16 frames
