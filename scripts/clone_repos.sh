set -u
BASE=~/claw_work/Research/YOLO-Mask-Selective/repos
clone() { # $1=url $2=dest
  dest="$BASE/$2"
  if [ -d "$dest/.git" ]; then echo "SKIP  $2 (exists)"; return; fi
  echo "CLONE $2 <- $1"
  git clone --depth 1 "$1" "$dest" 2>&1 | tail -2
}
clone https://github.com/jeromerony/alma_prox_segmentation.git            papers/alma_prox_segmentation
clone https://github.com/jeromerony/augmented_lagrangian_adversarial_attacks.git papers/alma_iccv2021
clone https://github.com/jeromerony/adversarial-library.git               papers/jeromerony_adversarial_library
clone https://github.com/ultralytics/ultralytics.git                     models/ultralytics
clone https://github.com/dbolya/yolact.git                               models/yolact
clone https://github.com/facebookresearch/detectron2.git                 models/detectron2
clone https://github.com/cocodataset/cocoapi.git                         tools/cocoapi
clone https://github.com/Harry24k/adversarial-attacks-pytorch.git        tools/adversarial-attacks-pytorch
clone https://github.com/anuragarnab/adversarial-attacks.git             tools/seg-robustness-cvpr2018
clone https://github.com/open-mmlab/mmsegmentation.git                   tools/mmsegmentation
clone https://github.com/open-mmlab/mmdetection.git                      tools/mmdetection
echo "ALL DONE"
