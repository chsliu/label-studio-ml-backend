REM =====
conda create --name openmmlab python=3.8 -y
conda activate openmmlab 

REM =====
REM Install the latest Label Studio ML SDK
REM https://github.com/HumanSignal/label-studio-ml-backend
REM =====
git clone https://github.com/HumanSignal/label-studio-ml-backend.git

REM =====
cd label-studio-ml-backend/
pip install -e .

REM =====
REM label-studio-ml-backend/
set root=%cd%

REM =====
REM https://mmdetection.readthedocs.io/en/latest/get_started.html
REM https://mmdetection.readthedocs.io/en/v1.2.0/INSTALL.html
REM =====
cd %root%
REM conda install -y pytorch torchvision -c pytorch
REM conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -U openmim
pip install chardet
REM pip install build-essential
pip install opencv-python-headless
mim install mmengine
mim install "mmcv>=2.0.0"
REM pip3 install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10/index.html
mim install mmdet
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .

REM =====
REM Verify the installation
REM =====
cd mmdetection/
python mmdet/utils/collect_env.py
mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest %root%/data/models/
python demo/image_demo.py demo/demo.jpg %root%/data/models/rtmdet_tiny_8xb32-300e_coco.py --weights %root%/data/models/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth --device cpu
REM python demo/image_demo.py demo/demo.jpg rtmdet_tiny_8xb32-300e_coco.py --weights rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth --device cuda
REM You will see a new image demo.jpg on your ./outputs/vis folder, where bounding boxes are plotted on cars, benches, etc.
REM outputs\vis\demo.jpg

REM =====
REM https://github.com/HumanSignal/label-studio/blob/master/docs/source/tutorials/object-detector.md#Installation
REM =====
cd %root%
pip install boto3
label-studio-ml init coco-detector --from label_studio_ml\examples\mmdetection\mmdetection.py --force
REM label-studio-ml init coco-detector --from label_studio_ml/examples/mmdetection/mmdetection.py --force

REM =====
REM download model
REM https://github.com/open-mmlab/mmdetection/tree/main/configs/faster_rcnn
REM ===== 
cd %root%/data/models/
REM curl https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth --output faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
REM mim download mmdet --config faster-rcnn_r50_fpn_1x_coco --dest .
REM mim download mmdet --config yolov3_mobilenetv2_8xb24-320-300e_coco --dest .
mim download mmdet --config yolox_tiny_8x8_300e_coco --dest .

REM ===== 
REM start ml backend
REM ===== 
cd %root%
call "start here.bat"
