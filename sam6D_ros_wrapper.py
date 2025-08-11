import os
import sys
import json
import time
import argparse
import numpy as np
import trimesh
import random
import importlib
from PIL import Image
import cv2
import logging
from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torchvision.transforms as T

# Add current directory to path for imports
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)
sys.path.append(os.path.join(code_dir, 'Instance_Segmentation_Model'))
sys.path.append(os.path.join(code_dir, 'Instance_Segmentation_Model', 'model'))
sys.path.append(os.path.join(code_dir, 'Instance_Segmentation_Model', 'utils'))
sys.path.append(os.path.join(code_dir, 'Instance_Segmentation_Model', 'provider'))
sys.path.append(os.path.join(code_dir, 'Instance_Segmentation_Model', 'segment_anything'))
sys.path.append(os.path.join(code_dir, 'Pose_Estimation_Model'))
sys.path.append(os.path.join(code_dir, 'Pose_Estimation_Model', 'provider'))
sys.path.append(os.path.join(code_dir, 'Pose_Estimation_Model', 'utils'))
sys.path.append(os.path.join(code_dir, 'Pose_Estimation_Model', 'model'))
sys.path.append(os.path.join(code_dir, 'Pose_Estimation_Model', 'model', 'pointnet2'))

import torch
import gorilla
import pycocotools.mask as cocomask

# ISM imports
from model.utils import Detections, convert_npz_to_json
from utils.bbox_utils import CropResizePad
from segment_anything.utils.amg import rle_to_mask, mask_to_rle_pytorch

# PEM imports
from data_utils import (
    load_im,
    get_bbox,
    get_point_cloud_from_depth,
    get_resize_rgb_choose,
)

import rospy
from geometry_msgs.msg import Pose, Point, Quaternion
import actionlib
from robokudo_msgs.msg import GenericImgProcAnnotatorResult, GenericImgProcAnnotatorAction
import ros_numpy
import transforms3d as tf3d
from sensor_msgs.msg import Image as ROSImage


"""
    This Script runs two ros-action-services; one for Instance Segmentation (ISM)
    and another for Pose Estimation (PEM). PEM depends strongly on the results of ISM, therefore
    they are not really separated as DOPE normally does it. ISM, gets called normally but does not
    return all information needed for PEM; the information is saved as a local variable. That means
    PEM has to access the latest ISM results directly; and can not function independently with another 
    ISM from DOPE.

    The implemented ISM is an advanced version of CNOS; ISM returns some masks and object detection 
    information and can maybe be used from another PEM (not tested).
"""


class SAM6D_ROS:
    def __init__(self, config_file):
        print(f"Using config file: {config_file}")
        with open(config_file, 'r') as f:
            config = json.load(f)

        self.object_name_mapping = config["object_mapping"]
        self.intrinsics = np.asarray(config['cam_K']).reshape((3, 3))
        print(f"Using intrinsics: {self.intrinsics}")
        
        # Load category mapping if available
        self.category_mapping = config.get("category_mapping", {})

        # Shared variables for ISM results
        self.latest_detection_ism = []
        self.latest_rgb_image = None
        self.latest_depth_image = None
        self.latest_image_dimensions = None

        # ISM Configuration - Choose between "sam" or "fastsam"
        # Load from config file if available, otherwise default to "fastsam"
        self.segmentor_model = config.get("segmentor_model", "fastsam")  # Default to FastSAM
        print(f"Using segmentor model: {self.segmentor_model}")
        
        if self.segmentor_model == "sam":
            self.ism_model_name = "ISM_sam"
            self.ism_config_path = os.path.join(code_dir, "Instance_Segmentation_Model", "configs", "model", "ISM_sam.yaml")
        elif self.segmentor_model == "fastsam":
            self.ism_model_name = "ISM_fastsam"
            self.ism_config_path = os.path.join(code_dir, "Instance_Segmentation_Model", "configs", "model", "ISM_fastsam.yaml")
        else:
            raise ValueError(f"Unsupported segmentor_model: {self.segmentor_model}. Use 'sam' or 'fastsam'.")

        # PEM Configuration
        self.pem_det_score_thresh = 0.4
        self.pem_gpus = "0"
        self.pem_model_name = "pose_estimation_model"
        self.pem_config_path = os.path.join(code_dir, "Pose_Estimation_Model", "config", "base.yaml")
        self.pem_iter = 600000
        self.pem_exp_id = 0

        # Set random seeds
        random.seed(42)
        torch.manual_seed(42)

        rospy.loginfo(f"Initializing SAM-6D ISM ({self.segmentor_model.upper()}) and PEM components")
        
        # Set CUDA device
        gorilla.utils.set_cuda_visible_devices(gpu_ids=self.pem_gpus)
        
        # Initialize both models
        self._initialize_ism_model()
        self._initialize_pem_model()
        
        # Store mesh information for PEM
        self.mesh_files = config.get('mesh_files', {})
        self.meshes = {}  # Store loaded meshes
        self.model_points = {}  # Store sampled model points for each object
        self.templates = {}  # Store templates for each object
        
        # RGB transform for preprocessing (shared between ISM and PEM)
        self.rgb_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # ISM Crop, resize, pad utility
        self.cropresize = CropResizePad(224)
        
        # Pre-load meshes and prepare model points for PEM
        self._load_meshes()
        
        # Load PEM configuration for test dataset
        self._load_pem_config()

        # Initialize ROS node and action servers
        rospy.init_node("sam6d_combined")
        
        # ISM Action Server
        self.ism_server = actionlib.SimpleActionServer('/object_detector/sam6dism',
                                                       GenericImgProcAnnotatorAction,
                                                       execute_cb=self.segment_objects,
                                                       auto_start=False)
        self.ism_server.start()
        
        # PEM Action Server
        self.pem_server = actionlib.SimpleActionServer('/pose_estimator/sam6dpem',
                                                       GenericImgProcAnnotatorAction,
                                                       execute_cb=self.estimate_pose,
                                                       auto_start=False)
        self.pem_server.start()
        
        print(f"Combined SAM-6D ISM ({self.segmentor_model.upper()}) and PEM is ready.")

    def _initialize_ism_model(self):
        """Initialize the SAM-6D ISM model with configurable segmentor (SAM or FastSAM)"""
        print(f"=> creating ISM model with {self.segmentor_model.upper()} ...")
        
        # Initialize Hydra configuration like in the template
        with initialize(version_base=None, config_path="Instance_Segmentation_Model/configs"):
            cfg = compose(config_name="run_inference.yaml")
            
        # Override model config based on segmentor choice
        if self.segmentor_model == "sam":
            with initialize(version_base=None, config_path="Instance_Segmentation_Model/configs/model"):
                cfg.model = compose(config_name='ISM_sam.yaml')
            # For SAM, you can set stability_score_thresh if needed
            # cfg.model.segmentor_model.stability_score_thresh = 0.95
        elif self.segmentor_model == "fastsam":
            with initialize(version_base=None, config_path="Instance_Segmentation_Model/configs/model"):
                cfg.model = compose(config_name='ISM_fastsam.yaml')
        else:
            raise ValueError(f"The segmentor_model {self.segmentor_model} is not supported!")
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        print("=> instantiating model...")
        ism_model = instantiate(cfg.model)
        
        # Setup device for different model components
        print(f"=> moving models to {self.device}...")
        ism_model.descriptor_model.model = ism_model.descriptor_model.model.to(self.device)
        ism_model.descriptor_model.model.device = self.device
        
        # Handle segmentor model setup (different for SAM vs FastSAM)
        if hasattr(ism_model.segmentor_model, "predictor"):
            # SAM model setup
            ism_model.segmentor_model.predictor.model = (
                ism_model.segmentor_model.predictor.model.to(self.device)
            )
        else:
            # FastSAM model setup
            ism_model.segmentor_model.model.setup_model(device=self.device, verbose=True)
        
        ism_model.eval()
        
        self.ism_model = ism_model
        self.ism_cfg = cfg
        
        print(f"=> ISM model initialization complete ({self.segmentor_model.upper()})")

    def _initialize_pem_model(self):
        """Initialize the SAM-6D PEM model"""
        print("=> creating PEM model ...")
        
        # Load configuration
        cfg = gorilla.Config.fromfile(self.pem_config_path)
        exp_name = self.pem_model_name + '_' + \
            os.path.splitext(self.pem_config_path.split("/")[-1])[0] + '_id' + str(self.pem_exp_id)
        log_dir = os.path.join("log", exp_name)

        cfg.exp_name = exp_name
        cfg.gpus = self.pem_gpus
        cfg.model_name = self.pem_model_name
        cfg.log_dir = log_dir
        cfg.test_iter = self.pem_iter

        # Import and create model
        MODEL = importlib.import_module(cfg.model_name)
        self.pem_model = MODEL.Net(cfg.model)
        self.pem_model = self.pem_model.cuda()
        self.pem_model.eval()
        
        # Load checkpoint
        checkpoint_path = os.path.join(code_dir, 'Pose_Estimation_Model', 'checkpoints', 'sam-6d-pem-base.pth')
        if os.path.exists(checkpoint_path):
            gorilla.solver.load_checkpoint(model=self.pem_model, filename=checkpoint_path)
            print(f"=> loaded checkpoint: {checkpoint_path}")
        else:
            rospy.logwarn(f"Checkpoint not found: {checkpoint_path}")
            
        self.pem_cfg = cfg
        print("=> PEM model initialization complete")

    def _load_pem_config(self):
        """Load test dataset configuration for PEM"""
        # Default configuration values
        self.pem_test_config = {
            'img_size': 224,
            'n_sample_template_point': 2048,
            'n_sample_observed_point': 1024,
            'n_sample_model_point': 5000,
            'n_template_view': 42,
            'rgb_mask_flag': True
        }
        
        # Try to load from model config if available
        if hasattr(self.pem_cfg, 'test_dataset'):
            for key in self.pem_test_config.keys():
                if hasattr(self.pem_cfg.test_dataset, key):
                    self.pem_test_config[key] = getattr(self.pem_cfg.test_dataset, key)

    def _load_meshes(self):
        """Pre-load meshes and sample model points for PEM"""
        for obj_name, mesh_file in self.mesh_files.items():
            if os.path.exists(mesh_file):
                mesh = trimesh.load(mesh_file)
                self.meshes[obj_name] = mesh
                
                # Sample model points (convert mm to meters like original SAM-6D)
                model_points = mesh.sample(self.pem_test_config['n_sample_model_point']).astype(np.float32) / 1000.0
                self.model_points[obj_name] = model_points
                
                rospy.loginfo(f"Loaded mesh for {obj_name}: {mesh_file}")
            else:
                rospy.logwarn(f"Mesh file not found for {obj_name}: {mesh_file}")

    # ISM Methods
    def _preprocess_image(self, image):
        """Preprocess image for ISM model"""
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Apply preprocessing
        rgb_tensor = self.rgb_transform(image).unsqueeze(0).to(self.device)
        return rgb_tensor, image

    def _run_detection(self, rgb_tensor, original_image):
        """Run object detection and segmentation with ISM"""
        print("=> running ISM inference...")
        
        with torch.no_grad():
            # Run model inference using the correct method for ISM
            detections = self.ism_model.segmentor_model.generate_masks(np.array(original_image))
            detections = Detections(detections)
            
            # Get descriptors for matching
            query_descriptors, query_appe_descriptors = self.ism_model.descriptor_model.forward(
                np.array(original_image), detections)
            
            # Match descriptors to get semantic scores
            (idx_selected_proposals, pred_idx_objects, 
             semantic_score, best_template) = self.ism_model.compute_semantic_score(query_descriptors)
            
            # Filter detections based on matching
            detections.filter(idx_selected_proposals)
            
            # Convert to dictionary format for easier processing
            detections_dict = {
                'masks': detections.masks,
                'scores': semantic_score,
                'labels': torch.ones(len(detections.masks)),  # Default to class 1
                'bboxes': detections.bboxes if hasattr(detections, 'bboxes') else []
            }
            
        return detections_dict

    def _convert_to_coco_format(self, detections, image_width, image_height):
        """Convert detections to COCO-style format matching ISM output"""
        results = []
        
        if detections is None:
            return results
            
        # Get detection components
        masks = detections.get('masks', [])
        scores = detections.get('scores', [])
        labels = detections.get('labels', [])
        bboxes = detections.get('bboxes', [])
        
        for i in range(len(masks)):
            if i >= len(scores) or i >= len(labels):
                continue
                
            mask = masks[i]
            score = float(scores[i])
            label = int(labels[i])
            
            # Convert mask to RLE format
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
                
            # Ensure mask is binary
            if mask.dtype != np.uint8:
                mask = (mask > 0.5).astype(np.uint8)
                
            # Convert to RLE format (similar to COCO)
            if isinstance(mask, np.ndarray):
                mask_tensor = torch.from_numpy(mask)
            else:
                mask_tensor = mask
                
            rle = mask_to_rle_pytorch(mask_tensor)
            
            # Get bounding box
            if i < len(bboxes):
                bbox = bboxes[i]
                if isinstance(bbox, torch.Tensor):
                    bbox = bbox.cpu().numpy()
                bbox_list = [int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])]
            else:
                # Compute bbox from mask
                y_indices, x_indices = np.where(mask > 0)
                if len(y_indices) > 0 and len(x_indices) > 0:
                    x_min, x_max = x_indices.min(), x_indices.max()
                    y_min, y_max = y_indices.min(), y_indices.max()
                    bbox_list = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
                else:
                    bbox_list = [0, 0, 0, 0]
            
            # Create detection entry
            detection_entry = {
                "scene_id": 0,
                "image_id": 0,
                "category_id": label,
                "bbox": bbox_list,
                "score": score,
                "time": 0.0,
                "segmentation": {
                    "counts": rle['counts'],
                    "size": [image_height, image_width]
                }
            }
            
            results.append(detection_entry)
            
        return results

    # PEM Methods
    def _get_template(self, template_path, tem_index=1):
        """Load a single template view for PEM"""
        rgb_path = os.path.join(template_path, f'rgb_{tem_index}.png')
        mask_path = os.path.join(template_path, f'mask_{tem_index}.png')
        xyz_path = os.path.join(template_path, f'xyz_{tem_index}.npy')

        if not all(os.path.exists(p) for p in [rgb_path, mask_path, xyz_path]):
            return None, None, None

        rgb = load_im(rgb_path).astype(np.uint8)
        xyz = np.load(xyz_path).astype(np.float32) / 1000.0  # Convert mm to meters like original SAM-6D
        mask = load_im(mask_path).astype(np.uint8) == 255

        bbox = get_bbox(mask)
        y1, y2, x1, x2 = bbox
        mask = mask[y1:y2, x1:x2]

        rgb = rgb[:,:,::-1][y1:y2, x1:x2, :]
        if self.pem_test_config['rgb_mask_flag']:
            rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)

        rgb = cv2.resize(rgb, (self.pem_test_config['img_size'], self.pem_test_config['img_size']), 
                        interpolation=cv2.INTER_LINEAR)
        rgb = self.rgb_transform(np.array(rgb))

        choose = (mask>0).astype(np.float32).flatten().nonzero()[0]
        if len(choose) <= self.pem_test_config['n_sample_template_point']:
            choose_idx = np.random.choice(np.arange(len(choose)), self.pem_test_config['n_sample_template_point'])
        else:
            choose_idx = np.random.choice(np.arange(len(choose)), self.pem_test_config['n_sample_template_point'], replace=False)
        choose = choose[choose_idx]
        xyz = xyz[y1:y2, x1:x2, :].reshape((-1, 3))[choose, :]

        rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], self.pem_test_config['img_size'])
        return rgb, rgb_choose, xyz

    def _get_templates(self, template_path):
        """Load all template views for an object for PEM"""
        n_template_view = self.pem_test_config['n_template_view']
        all_tem = []
        all_tem_choose = []
        all_tem_pts = []

        total_nView = 42
        for v in range(n_template_view):
            i = int(total_nView / n_template_view * v)
            tem, tem_choose, tem_pts = self._get_template(template_path, i)
            if tem is not None:
                all_tem.append(torch.FloatTensor(tem).unsqueeze(0).cuda())
                all_tem_choose.append(torch.IntTensor(tem_choose).long().unsqueeze(0).cuda())
                all_tem_pts.append(torch.FloatTensor(tem_pts).unsqueeze(0).cuda())

        if not all_tem:
            return None, None, None
            
        return all_tem, all_tem_pts, all_tem_choose

    def _extract_template_features(self, obj_name, template_path):
        """Extract template features for an object for PEM"""
        if obj_name in self.templates:
            return self.templates[obj_name]
            
        print(f"=> extracting templates for {obj_name}...")
        all_tem, all_tem_pts, all_tem_choose = self._get_templates(template_path)
        
        if all_tem is None:
            rospy.logwarn(f"No templates found for {obj_name} at {template_path}")
            return None, None
            
        with torch.no_grad():
            all_tem_pts_feat, all_tem_feat = self.pem_model.feature_extraction.get_obj_feats(
                all_tem, all_tem_pts, all_tem_choose)
        
        # Store templates
        self.templates[obj_name] = (all_tem_pts_feat, all_tem_feat)
        return all_tem_pts_feat, all_tem_feat

    def _process_segmentation(self, mask, rgb_img, depth_img, obj_name):
        """Process segmentation mask and extract features for PEM"""
        # Get whole image point cloud
        whole_pts = get_point_cloud_from_depth(depth_img, self.intrinsics)
        
        # Get model points for this object
        model_points = self.model_points[obj_name]
        radius = np.max(np.linalg.norm(model_points, axis=1))

        # Process mask
        mask = np.logical_and(mask > 0, depth_img > 0)
        if np.sum(mask) <= 32:
            return None
            
        bbox = get_bbox(mask)
        y1, y2, x1, x2 = bbox
        mask = mask[y1:y2, x1:x2]
        choose = mask.astype(np.float32).flatten().nonzero()[0]

        # Process point cloud
        cloud = whole_pts.copy()[y1:y2, x1:x2, :].reshape(-1, 3)[choose, :]
        center = np.mean(cloud, axis=0)
        tmp_cloud = cloud - center[None, :]
        flag = np.linalg.norm(tmp_cloud, axis=1) < radius * 1.2
        if np.sum(flag) < 4:
            return None
            
        choose = choose[flag]
        cloud = cloud[flag]

        if len(choose) <= self.pem_test_config['n_sample_observed_point']:
            choose_idx = np.random.choice(np.arange(len(choose)), self.pem_test_config['n_sample_observed_point'])
        else:
            choose_idx = np.random.choice(np.arange(len(choose)), self.pem_test_config['n_sample_observed_point'], replace=False)
        choose = choose[choose_idx]
        cloud = cloud[choose_idx]

        # Process RGB
        rgb = rgb_img.copy()[y1:y2, x1:x2, :][:,:,::-1]
        if self.pem_test_config['rgb_mask_flag']:
            rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)
        rgb = cv2.resize(rgb, (self.pem_test_config['img_size'], self.pem_test_config['img_size']), 
                        interpolation=cv2.INTER_LINEAR)
        rgb = self.rgb_transform(np.array(rgb))
        rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], self.pem_test_config['img_size'])

        return {
            'rgb': torch.FloatTensor(rgb),
            'cloud': torch.FloatTensor(cloud),
            'rgb_choose': torch.IntTensor(rgb_choose).long(),
            'model_points': model_points
        }

    def _convert_ism_to_masks(self, ism_detections, image_shape):
        """Convert ISM detections to mask format for PEM"""
        masks = []
        class_names = []
        scores = []
        
        for detection in ism_detections:
            try:
                # Extract segmentation RLE
                rle = detection['segmentation']
                
                # Convert RLE to mask
                mask = cocomask.decode(rle)
                
                # Get class information
                category_id = detection.get('category_id', 1)
                score = detection.get('score', 1.0)
                
                # Map category to class name - use a more robust mapping
                class_name = None
                
                # Try to map based on category_id if available in config
                if hasattr(self, 'category_mapping') and str(category_id) in self.category_mapping:
                    class_name = self.category_mapping[str(category_id)]
                else:
                    # Default mapping - use first available object or a default name
                    object_names = list(self.object_name_mapping.keys())
                    if object_names:
                        class_name = object_names[0]  # Use first object as default
                    else:
                        class_name = f"category_{category_id}"
                
                masks.append(mask)
                class_names.append(class_name)
                scores.append(score)
                
            except Exception as e:
                print(f"  -> Failed to process detection: {str(e)}")
                continue
                
        return masks, class_names, scores

    # Action Server Callbacks
    def segment_objects(self, req):
        """ISM Action Server Callback"""
        print("ISM: Request segmentation...")
        start_time = time.time()

        # Extract data from request
        rgb = req.rgb
        
        # Get image dimensions
        width, height = rgb.width, rgb.height
        print(f"Image dimensions: {width}x{height}")
        
        # Convert ROS message to numpy array
        image = ros_numpy.numpify(rgb)
        
        # Ensure RGB image is in correct format (H, W, 3) with uint8
        if len(image.shape) == 3 and image.shape[2] == 3:
            if image.dtype != np.uint8:
                if image.max() <= 1.0:  # Normalized to [0,1]
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
        else:
            print(f"WARNING: Unexpected RGB image format: {image.shape}, dtype: {image.dtype}")
        
        print("RGB", image.shape, image.dtype)
        
        # Store image data for PEM usage
        self.latest_rgb_image = image.copy()
        self.latest_image_dimensions = (width, height)
        
        # Also get depth if available for PEM
        try:
            depth_img = ros_numpy.numpify(req.depth)
            if depth_img.dtype == np.uint16:
                depth_img = depth_img.astype(np.float32) / 1000.0
            elif depth_img.dtype != np.float32:
                depth_img = depth_img.astype(np.float32)
            self.latest_depth_image = depth_img.copy()
            print("Depth image: Available and stored")
        except:
            self.latest_depth_image = None
            print("Depth image: Not available")
        
        try:
            # Preprocess image
            rgb_tensor, original_image = self._preprocess_image(image)
            
            # Run detection
            detections = self._run_detection(rgb_tensor, original_image)
            
            # Convert to COCO format
            results = self._convert_to_coco_format(detections, width, height)
            
            # Store results in shared variable for PEM
            self.latest_detection_ism = results.copy()
            
            print(f"=> processed {len(results)} detections")
            
            # Select best detection for response (highest score)
            if results:
                best_detection = max(results, key=lambda x: x['score'])
                results_to_return = [best_detection]
                print(f"=> returning best detection with score: {best_detection['score']:.3f}")
            else:
                results_to_return = []
            
            # Convert detections to ROS format for response
            response_masks = []
            response_class_names = []
            
            for detection in results_to_return:
                # Convert RLE back to mask for ROS response
                rle = detection['segmentation']
                mask = rle_to_mask(rle)
                
                # Convert to ROS Image message
                mask_msg = ros_numpy.msgify(ROSImage, mask.astype(np.uint8), encoding='mono8')
                response_masks.append(mask_msg)
                
                # Map category_id back to class name
                category_id = detection['category_id']
                class_name = f"object_{category_id}"  # Default naming
                
                # Try to find corresponding class name from mapping
                for obj_name, mapped_name in self.object_name_mapping.items():
                    if category_id == 1:  # Assuming category_id mapping logic
                        class_name = obj_name
                        break
                        
                response_class_names.append(class_name)
                
            print(f"=> returning {len(response_masks)} detections to ROS")
            
        except Exception as e:
            print(f"=> Segmentation failed: {str(e)}")
            rospy.logwarn(f"Segmentation failed: {str(e)}")
            response_masks = []
            response_class_names = []
            self.latest_detection_ism = []

        # Create response
        response = GenericImgProcAnnotatorResult()
        response.mask_detections = response_masks
        response.class_names = response_class_names
        response.pose_results = []  # ISM doesn't provide poses

        end_time = time.time()
        elapsed_time = end_time - start_time
        print('ISM Execution time:', elapsed_time, 'seconds')
        
        self.ism_server.set_succeeded(response)

    def estimate_pose(self, req):
        """PEM Action Server Callback - Uses shared ISM results"""
        print("PEM: Request pose estimation...")
        start_time = time.time()

        # Use shared ISM results instead of request data
        if not self.latest_detection_ism:
            print("PEM: No ISM detections available!")
            response = GenericImgProcAnnotatorResult()
            response.pose_results = []
            response.class_names = []
            self.pem_server.set_succeeded(response)
            return

        if self.latest_rgb_image is None:
            print("PEM: No RGB image available!")
            response = GenericImgProcAnnotatorResult()
            response.pose_results = []
            response.class_names = []
            self.pem_server.set_succeeded(response)
            return

        if self.latest_depth_image is None:
            print("PEM: No depth image available!")
            response = GenericImgProcAnnotatorResult()
            response.pose_results = []
            response.class_names = []
            self.pem_server.set_succeeded(response)
            return

        print(f"PEM: Using {len(self.latest_detection_ism)} ISM detections")
        
        # Convert ISM detections to masks and class names
        width, height = self.latest_image_dimensions
        mask_detections, class_names, scores = self._convert_ism_to_masks(
            self.latest_detection_ism, (height, width))

        print(f"PEM: Converted to {len(mask_detections)} masks")
        for i, name in enumerate(class_names):
            print(f"  Object {i+1}: {name}")
            if name in self.object_name_mapping:
                mapped_name = self.object_name_mapping[name]
                print(f"    -> Mapped to: {mapped_name}")
                if mapped_name in self.meshes:
                    print(f"    -> Mesh available: YES")
                else:
                    print(f"    -> Mesh available: NO")
            else:
                print(f"    -> Not found in object mapping")

        # Process each detected object
        valid_class_names = []
        pose_results = []

        # Collect all valid detections for batch processing
        batch_data = []
        batch_objects = []
        
        for i, class_name in enumerate(class_names):
            print(f"PEM: Processing object {i+1}/{len(class_names)}: {class_name}")
            
            # Check if we have a mapping for this class
            if class_name not in self.object_name_mapping:
                print(f"  -> No mapping found for {class_name}, skipping")
                continue
                
            mapped_name = self.object_name_mapping[class_name]
            print(f"  -> Mapped to: {mapped_name}")
            
            # Check if we have mesh for this object
            if mapped_name not in self.meshes:
                print(f"  -> No mesh available for {mapped_name}, skipping")
                continue
                
            # Check if we have corresponding mask
            if i >= len(mask_detections):
                print(f"  -> No mask available for object {i}, skipping")
                continue
                
            mask = mask_detections[i]
            
            # Debug mask information
            mask_pixels = np.sum(mask > 0)
            print(f"  -> Mask has {mask_pixels} non-zero pixels")
            if mask_pixels == 0:
                print(f"  -> Empty mask for {class_name}, skipping")
                continue

            # Process segmentation
            processed_data = self._process_segmentation(mask, self.latest_rgb_image, self.latest_depth_image, mapped_name)
            if processed_data is None:
                print(f"  -> Failed to process segmentation for {class_name}")
                continue

            # Get templates for this object - extract object ID from mapped_name
            # Convert mapped name like "006_mustard_bottle" to object ID like "obj_000005"
            template_path = None
            if mapped_name in self.mesh_files:
                mesh_file = self.mesh_files[mapped_name]
                # Extract object ID from mesh file path (e.g., obj_000005.ply -> obj_000005)
                obj_id = os.path.splitext(os.path.basename(mesh_file))[0]
                # Try multiple possible template paths
                possible_paths = [
                    os.path.join(code_dir, "Data", "Example", "outputs", "templates"),
                    os.path.join("/code", "templates", obj_id),
                    os.path.join(code_dir, "templates", obj_id),
                    os.path.join(code_dir, "templates", mapped_name)
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        template_path = path
                        break
            
            if template_path is None or not os.path.exists(template_path):
                print(f"  -> Template path not found for {mapped_name}. Tried paths:")
                for path in possible_paths:
                    print(f"     {path}")
                continue

            template_features = self._extract_template_features(mapped_name, template_path)
            if template_features is None:
                print(f"  -> Failed to extract template features for {mapped_name}")
                continue

            batch_data.append(processed_data)
            batch_objects.append((class_name, mapped_name, template_features))

        if not batch_data:
            print("PEM: No valid objects to process")
            response = GenericImgProcAnnotatorResult()
            response.pose_results = []
            response.class_names = []
            self.pem_server.set_succeeded(response)
            return

        # Process objects individually since they may be different types
        for i, (class_name, mapped_name, template_features) in enumerate(batch_objects):
            try:
                print(f"PEM: Processing object {i+1}/{len(batch_objects)}: {class_name}")
                
                # Prepare single object input
                single_data = batch_data[i]
                rgb_tensor = single_data['rgb'].unsqueeze(0).cuda()
                cloud_tensor = single_data['cloud'].unsqueeze(0).cuda()
                rgb_choose_tensor = single_data['rgb_choose'].unsqueeze(0).cuda()
                model_points = single_data['model_points']
                
                model_tensor = torch.FloatTensor(model_points).unsqueeze(0).cuda()
                K_tensor = torch.FloatTensor(self.intrinsics).unsqueeze(0).cuda()
                
                input_data = {
                    'pts': cloud_tensor,
                    'rgb': rgb_tensor,
                    'rgb_choose': rgb_choose_tensor,
                    'model': model_tensor,
                    'K': K_tensor
                }

                with torch.no_grad():
                    # Use template features for this specific object
                    all_tem_pts, all_tem_feat = template_features
                    input_data['dense_po'] = all_tem_pts
                    input_data['dense_fo'] = all_tem_feat
                    
                    out = self.pem_model(input_data)

                # Process results
                if 'pred_pose_score' in out.keys():
                    pose_score = out['pred_pose_score'][0]
                else:
                    pose_score = 1.0
                    
                pose_score = float(pose_score.detach().cpu().numpy())
                pred_rot = out['pred_R'][0].detach().cpu().numpy()
                pred_trans = out['pred_t'][0].detach().cpu().numpy() * 1000  # Convert back to mm

                # Convert pose to ROS Pose message
                pose_msg = Pose()
                
                rotation_matrix = pred_rot
                translation = pred_trans
                
                # Check if pose is valid
                if np.allclose(translation, 0) and np.allclose(rotation_matrix, np.eye(3)):
                    print(f"  -> WARNING: Invalid pose for {class_name}")
                    continue
                
                # Convert rotation matrix to quaternion
                quaternion = tf3d.quaternions.mat2quat(rotation_matrix)

                pose_msg.position = Point(x=float(translation[0]), 
                                        y=float(translation[1]), 
                                        z=float(translation[2]))
                pose_msg.orientation = Quaternion(x=float(quaternion[1]), 
                                                y=float(quaternion[2]), 
                                                z=float(quaternion[3]), 
                                                w=float(quaternion[0]))
                
                print(f"  -> Pose for {class_name}:")
                print(f"     Position: [{translation[0]:.6f}, {translation[1]:.6f}, {translation[2]:.6f}]")
                print(f"     Score: {pose_score:.6f}")
                
                pose_results.append(pose_msg)
                valid_class_names.append(class_name)
                
            except Exception as e:
                print(f"  -> Error processing {class_name}: {str(e)}")
                continue

        print("PEM: => processing complete")
        
        # Create response
        response = GenericImgProcAnnotatorResult()
        response.pose_results = pose_results
        response.class_names = valid_class_names

        end_time = time.time()
        elapsed_time = end_time - start_time
        print('PEM Execution time:', elapsed_time, 'seconds')
        
        self.pem_server.set_succeeded(response)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', 
                       default="./configs/cfg_ros_ycbv_inference.json",
                       help='Path to configuration file')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    sam6d_ros = SAM6D_ROS(**vars(opt))
    
    rospy.spin()
