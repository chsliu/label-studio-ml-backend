REM https://labelstud.io/tutorials/object-detector#GPU-support

set config_file=%cd%\mmdetection\configs\faster_rcnn\faster-rcnn_r50_fpn_1x_coco.py
set checkpoint_file=%cd%\faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
set LABEL_STUDIO_HOSTNAME=http://localhost:8080
REM export config_file=mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py
REM export checkpoint_file=faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth

REM conda activate openmmlab && label-studio-ml start coco-detector --with config_file=%cd%\mmdetection\configs\faster_rcnn\faster-rcnn_r50_fpn_1x_coco.py checkpoint_file=%cd%\faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth device=gpu:0

REM label-studio-ml start coco-detector --with device=gpu:0 --kwargs hostname=http://localhost:8080
conda activate openmmlab && label-studio-ml start coco-detector --with device=gpu:0

REM label-studio-ml start coco-detector --with config_file=mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py checkpoint_file=faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth device=gpu:0

REM docker run -it --rm -v %cd%\mmdetection:/mmdetection humansignal/ml-backend:v0 bash

REM docker-compose up

REM http://host.docker.internal:9090

pause
