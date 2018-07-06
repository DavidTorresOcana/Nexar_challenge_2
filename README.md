# Nexar_challenge_2
Nexar challenge #2 https://www.getnexar.com/challenge-2/

## How to train
Use the notebook "Nexar YOLO Train-eval"

## Models trained and results
Yolov2 was chosen as base architecture:
* Square input 416x416: Best performance achieved ~70% mAP@0.5IOU
     - Anchors used: yolo_anchors.txt
     - Output: 13x13x((1+4+5)*5)
* Wide input 416x608: Best performance achieved ~65% mAP@0.5IOU
    - Anchors used: yolo_wide_nexar_anchors.txt
    - Output: 13x19x((1+4+5)*5)

