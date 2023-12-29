@echo off

REM =====
REM https://labelstud.io/tutorials/object-detector#GPU-support

REM =====
set LABEL_STUDIO_HOST=http://localhost:8080
set LABEL_STUDIO_API_KEY=b7e78c338c72f337a00df7633d81a64138f7eee8

REM =====
set config_file=%cd%\data\models\faster-rcnn_r50_fpn_1x_coco.py
set checkpoint_file=%cd%\data\models\faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
REM =====
REM set config_file=%cd%\data\models\yolov3_mobilenetv2_8xb24-320-300e_coco.py
REM set checkpoint_file=%cd%\data\models\yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth
REM =====
REM set config_file=%cd%\data\models\yolox_tiny_8x8_300e_coco.py
REM set checkpoint_file=%cd%\data\models\yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth

REM =====
REM export config_file=mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py
REM export checkpoint_file=faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth

REM =====
REM conda activate openmmlab && label-studio-ml start coco-detector --with config_file=%cd%\mmdetection\configs\faster_rcnn\faster-rcnn_r50_fpn_1x_coco.py checkpoint_file=%cd%\faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth device=gpu:0

REM =====
REM label-studio-ml start coco-detector --with device=gpu:0 --kwargs hostname=http://localhost:8080

REM =====
conda activate openmmlab && label-studio-ml start coco-detector --with device=gpu:0

REM =====
REM label-studio-ml start coco-detector --with config_file=mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py checkpoint_file=faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth device=gpu:0

REM =====
REM docker-compose up

REM =====
REM docker run -it --rm -v %cd%\mmdetection:/mmdetection humansignal/ml-backend:v0 bash
REM docker run -it --rm -v ${pwd}\mmdetection:/mmdetection humansignal/ml-backend:v0 bash

REM =====
REM http://localhost:8080
REM http://host.docker.internal:9090

pause
