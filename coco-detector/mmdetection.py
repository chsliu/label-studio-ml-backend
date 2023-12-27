import os
import logging
import boto3
import io
import json


from mmdet.apis import init_detector, inference_detector

from label_studio_ml.model import LabelStudioMLBase, update_fn
from label_studio_ml.utils import get_image_size, \
    get_single_tag_keys, DATA_UNDEFINED_NAME
from label_studio_tools.core.utils.io import get_data_dir
from botocore.exceptions import ClientError
from urllib.parse import urlparse


logger = logging.getLogger(__name__)
access_token = 'b7e78c338c72f337a00df7633d81a64138f7eee8'

class MMDetection(LabelStudioMLBase):
    """Object detector based on https://github.com/open-mmlab/mmdetection"""

    def __init__(self, config_file=None,
                 checkpoint_file=None,
                 image_dir=None,
                 labels_file=None, score_threshold=0.3, device='cpu', **kwargs):
        """
        Load MMDetection model from config and checkpoint into memory.
        (Check https://mmdetection.readthedocs.io/en/v1.2.0/GETTING_STARTED.html#high-level-apis-for-testing-images)

        Optionally set mappings from COCO classes to target labels
        :param config_file: Absolute path to MMDetection config file (e.g. /home/user/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x.py)
        :param checkpoint_file: Absolute path MMDetection checkpoint file (e.g. /home/user/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth)
        :param image_dir: Directory where images are stored (should be used only in case you use direct file upload into Label Studio instead of URLs)
        :param labels_file: file with mappings from COCO labels to custom labels {"airplane": "Boeing"}
        :param score_threshold: score threshold to wipe out noisy results
        :param device: device (cpu, cuda:0, cuda:1, ...)
        :param kwargs: can contain endpoint_url in case of non amazon s3
        """
        super(MMDetection, self).__init__(**kwargs)
        
        config_file = None
        
        # print("=====")
        # print("config_file:",config_file)
        # print("checkpoint_file:",checkpoint_file)
        # print("kwargs:",kwargs)
        # print("LABEL_STUDIO_HOSTNAME:",os.environ['LABEL_STUDIO_HOSTNAME'])
        # print("labels_file:",labels_file)
        # print("=====")
        
        config_file = config_file or os.environ['config_file']
        checkpoint_file = checkpoint_file or os.environ['checkpoint_file']
        self.hostname = os.environ['LABEL_STUDIO_HOSTNAME']
        self.access_token = access_token
        update_fn(fit_helper)
        
        # init_log()
        # find_log()
        
        # print("=====")
        # print("os.environ['config_file']:",os.environ['config_file'])
        # print("os.environ['checkpoint_file']:",os.environ['checkpoint_file'])
        # print("self.parsed_label_config:",self.parsed_label_config)
        # self.get('parsed_label_config')
        # print("parsed_label_config:",self.get('parsed_label_config'))
        # print('Load new model from: ', config_file, checkpoint_file)
        # self.model = init_detector(config_file, checkpoint_file, device=device)
        # print("=====")
        
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.labels_file = labels_file
        self.endpoint_url = kwargs.get('endpoint_url')
        if self.endpoint_url:
            logger.info(f'Using s3 endpoint url {self.endpoint_url}')
        
        # default Label Studio image upload folder
        upload_dir = os.path.join(get_data_dir(), 'media', 'upload')
        self.image_dir = image_dir or upload_dir
        logger.debug(f'{self.__class__.__name__} reads images from {self.image_dir}')
        if self.labels_file and os.path.exists(self.labels_file):
            self.label_map = json_load(self.labels_file)
        else:
            self.label_map = {}

        # print("=====")
        # print(dir(self))
        # print(type(self.parsed_label_config))
        # print("=====")
        
        if not self.get('parsed_label_config'):
            self.set('parsed_label_config',"{}")
        
        if len(self.parsed_label_config)==1:
            self.from_name, self.to_name, self.value, self.labels_in_config = get_single_tag_keys(
                self.parsed_label_config, 'RectangleLabels', 'Image')
            schema = list(self.parsed_label_config.values())[0]
            self.labels_in_config = set(self.labels_in_config)

            # Collect label maps from `predicted_values="airplane,car"` attribute in <Label> tag
            self.labels_attrs = schema.get('labels_attrs')
            if self.labels_attrs:
                for label_name, label_attrs in self.labels_attrs.items():
                    for predicted_value in label_attrs.get('predicted_values', '').split(','):
                        self.label_map[predicted_value] = label_name

        print('Load new model from: ', config_file, checkpoint_file)
        self.model = init_detector(config_file, checkpoint_file, device=device)
        self.score_thresh = score_threshold
        
        # print("=====")
        # print("self.model:",self.model)
        # print("self.model._version:",self.model._version)
        # print("self.model._version:",self.model._version())
        # print("self.model._get_name:",self.model._get_name)
        # print("self.model._get_name:",self.model._get_name())
        # print("dir(self.model):",dir(self.model))
        # for m in self.label_map:
            # print("self.label_map:", m)
        # print("=====")

    def _get_image_url(self, task):
        image_url = task['data'].get(self.value) or task['data'].get(DATA_UNDEFINED_NAME)
        if image_url.startswith('s3://'):
            # presign s3 url
            r = urlparse(image_url, allow_fragments=False)
            bucket_name = r.netloc
            key = r.path.lstrip('/')
            client = boto3.client('s3', endpoint_url=self.endpoint_url)
            try:
                image_url = client.generate_presigned_url(
                    ClientMethod='get_object',
                    Params={'Bucket': bucket_name, 'Key': key}
                )
            except ClientError as exc:
                logger.warning(f'Can\'t generate presigned URL for {image_url}. Reason: {exc}')
        return image_url

    def predict(self, tasks, **kwargs):
        assert len(tasks) == 1
        task = tasks[0]
        
        # print("=====")
        image_url = self._get_image_url(task)
        # print("image_url:",image_url)
        image_path = self.get_local_path(image_url)
        # print("=====")
        
        # print("=====")
        # print("self.model:",self.model)
        # print("self.labels_in_config:",self.labels_in_config)
        # print("self.label_map:",self.label_map)
        # print("type(self.label_map):", type(self.label_map))
        # print("dir(self.label_map):", dir(self.label_map))
        # for m in self.label_map:
            # print("type(m):", type(m))
            # print("dir(m):", dir(m))
            # print("self.label_map:", m,self.label_map[m])
        # classes = self.model.dataset_meta.get('classes')
        # print("classes(%d): %s"%(len(classes),sorted(classes)))
        # print("=====")
        
        self.model.CLASSES = self.model.dataset_meta.get('classes')
        
        model_results = inference_detector(self.model, image_path).pred_instances
        
        # print("=====")
        # print("dir(model_results):",dir(model_results))
        # print("model_results:",model_results)
        # print("model_results(%d)"%(len(model_results)))
        # c = 0
        # for i in zip(model_results):
            # print("model_results[%d]: %s"%(c,i))
            # c = c+1
        # print("=====")
        results = []
        all_scores = []
        img_width, img_height = get_image_size(image_path)
        # for item, label in zip(model_results, self.model.CLASSES):
        for item in model_results:
            bboxes, label_id, scores = item['bboxes'], item['labels'][0], item['scores']
            label_id = int(label_id)
            scores = list(scores)
            label = self.model.CLASSES[label_id]
            output_label = self.label_map.get(label, label)
            
            # print("=====")
            # print("label:",label)
            # print("output_label:",output_label)
            # for m in self.label_map:
                # print("self.label_map[%s]=%s"%(m,self.label_map[m]))
            # print("=====")

            # bboxes, label_id, scores = item['bboxes'], item['labels'][0], item['scores']
            
            # print("label:",label)
            # print("=====")
            # label_id = int(label_id)
            # scores = list(scores)
            
            # print("=====")
            # print("item:",item)
            # print("type(bboxes):", type(bboxes))
            # print("dir(bboxes):", dir(bboxes))
            # print("dir(label):",dir(label))
            # print("type(label):",type(label))
            # print("label:",label)
            # print("label_id:",label_id)
            # print("scores:",scores)
            # print("type(scores):",type(bboxes))
            # for bb in bboxes:
                # print("type(bb):", type(bb))
                # print("dir(bb):", dir(bb))
                # print("bboxes:", bb)
            # print("=====")
            
            # print("=====")
            if output_label not in self.labels_in_config:
                print(output_label + ' label not found in project config.')
                continue
            # else:
                # print("=== %s label founded"%(output_label))
            # print("=====")
            for bbox in bboxes:
            
                # print("=====")
                # print("item:",item)
                # print("type(bbox):", type(bbox))
                # print("dir(bbox):", dir(bbox))
                # print("bbox:", bbox)
                # bb = list(bbox)
                # print("type(bb):", type(bb))
                # print("bb:", bb)
                # score = float(bb[-1])
                # print("score:", score)
                # print("=====")
                
                # print("=====")
                # print("self.label_map:",self.label_map)
                # print("type(self.label_map):",type(self.label_map))
                # for m in self.label_map:
                    # print("self.label_map[%s]=%s"%(m,self.label_map[m]))
                # print("=====")
        
                bbox = list(bbox)
                if not bbox:
                    print("not bbox")
                    continue
                # score = float(bbox[-1])
                score = float(scores.pop(0))
                # print("=====")
                # print("label_id:",label_id)
                # print("bbox:", bbox)
                # print("score:", score)
                # print("=====")
                if score < self.score_thresh:
                    continue
                # x, y, xmax, ymax = bbox[:4]
                x, y, xmax, ymax = [float(i) for i in bbox]
                # print("=====")
                # print("x:", x)
                # print("y:", y)
                # print("xmax:", xmax)
                # print("ymax:", ymax)
                # print("=====")
                results.append({
                    'from_name': self.from_name,
                    'to_name': self.to_name,
                    'type': 'rectanglelabels',
                    'value': {
                        'rectanglelabels': [output_label],
                        'x': x / img_width * 100,
                        'y': y / img_height * 100,
                        'width': (xmax - x) / img_width * 100,
                        'height': (ymax - y) / img_height * 100
                    },
                    'score': score
                })
                all_scores.append(score)
        avg_score = sum(all_scores) / max(len(all_scores), 1)
        # print("=====")
        # print("results:",results)
        # print("avg_score:",avg_score)
        # print("=====")
        return [{
            'result': results,
            'score': avg_score
        }]

    def fit(self, event, data,  **kwargs):
        """
        This method is called each time an annotation is created or updated
        It simply stores the latest annotation as a "prediction model" artifact
        """
        # print("=====")
        # print("type(event):",type(event))
        # print("=====")
        print("MMDetection::fit(%s, data, %s) is called" %(event,kwargs))
        
        if event.startswith('ANNOTATION_UPDATED'):
            # print("=====")
            # print("data:",data)
            # print("=====")
            self.set('last_annotation', json.dumps(data['annotation']['result']))
        elif event.startswith('PROJECT_UPDATED'):
            # print("=====")
            # print("data:",data)
            # print("=====")
            self.set('last_project', json.dumps(data['project']))
            
        # to control the model versioning, you can use the model_version parameter
        # self.set('model_version', str(uuid4())[:8])
        self.set('model_version', self.model._get_name())


def json_load(file, int_keys=False):
    with io.open(file, encoding='utf8') as f:
        data = json.load(f)
        if int_keys:
            return {int(k): v for k, v in data.items()}
        else:
            return data


def fit_helper(event, data, helper, **additional_params):
    helper.fit(event, data, additional_params)
    

def init_log():
    import logging
    import sys

    root = logging.getLogger()
    # root.setLevel(logging.DEBUG)
    root.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    # handler.setLevel(logging.DEBUG)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)

    
def find_log():
    import logging
    from logging import FileHandler

    # note, this will create a new logger if the name doesn't exist, 
    # which will have no handlers attached (yet)
    logger = logging.getLogger('<name>')

    # print("find_log")

    for h in logger.handlers:
        # check the handler is a file handler 
        # (rotating handler etc. inherit from this, so it will still work)
        # stream handlers write to stderr, so their filename is not useful to us
        if isinstance(h, FileHandler):
            # h.stream should be an open file handle, it's name is the path
            print(h.stream.name)