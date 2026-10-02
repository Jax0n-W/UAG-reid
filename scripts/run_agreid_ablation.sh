#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:?Set DATA_DIR to the prepared AG-ReID root}"
LOGS_DIR="${LOGS_DIR:-logs}"
PRETRAINED_RESNET50="${PRETRAINED_RESNET50:?Set PRETRAINED_RESNET50 to resnet50-19c8e357.pth}"

COMMON_ARGS=(
  --dataset agreid_ir
  --data-dir "$DATA_DIR"
  --logs-dir "$LOGS_DIR"
  --pretrained-resnet50 "$PRETRAINED_RESNET50"
  --arch agw
  --memorybank CMhybrid
  --height 288
  --width 144
  --batch-size 64
  --num-instances 16
  --epochs 50
  --iters 400
  --lr 0.00035
  --weight-decay 0.0005
  --momentum 0.2
  --step-size 20
  --eps 0.6
  --eps-gap 0.02
  --k1 30
  --k2 6
  --temp 0.05
  --seed 1
  --trial 1
  --workers 8
  --pooling-type gem
  --eval-step 1
  --eval-during-train=False
  --rahp-beta 0.25
  --rahp-knn 20
  --rahp-alpha 0.5
  --cesa-rho 0.8
  --cesa-eta 0.1
  --cesa-lineage-thr 0.5
  --cesa-warmup 5
)

run_experiment() {
  local name="$1"
  shift
  python train_agreid.py "${COMMON_ARGS[@]}" --logs-dir "$LOGS_DIR/$name" "$@"
}

run_experiment baseline
run_experiment rahp --use-rahp
run_experiment cesa --use-cesa
run_experiment full --use-rahp --use-cesa
