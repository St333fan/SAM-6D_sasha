import os
import sys
import json
import time
import argparse
import numpy as np
import torch
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

from model.utils import Detections, convert_npz_to_json
from utils.bbox_utils import CropResizePad
from segment_anything.utils.amg import rle_to_mask

import rospy
from geometry_msgs.msg import Pose, Point, Quaternion
import actionlib
from robokudo_msgs.msg import GenericImgProcAnnotatorResult, GenericImgProcAnnotatorAction
import ros_numpy
from sensor_msgs.msg import Image as ROSImage


class SAM6DISM_ROS:
    def __init__(self, config_file, output_dir="/tmp/sam6d_ism"):
        print(f"Using config file: {config_file}")
        with open(config_file, 'r') as f:
            config = json.load(f)

        self.object_name_mapping = config["object_mapping"]
        self.intrinsics = np.asarray(config['cam_K']).reshape((3, 3))
        self.output_dir = output_dir
        print(f"Using intrinsics: {self.intrinsics}")
        print(f"Output directory: {self.output_dir}")

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Configuration parameters for SAM-6D ISM
        self.model_name = "ISM_sam"  # or ISM_fastsam
        self.config_path = os.path.join(code_dir, "Instance_Segmentation_Model", "configs", "model", "ISM_sam.yaml")
        
        rospy.loginfo("Initializing SAM-6D ISM components")
        
        # Initialize SAM-6D ISM model
        self._initialize_model()
        
        # Store mesh information for each object (needed for detection filtering)
        self.mesh_files = config.get('mesh_files', {})
        
        # Preprocessing transforms
        self.rgb_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Crop, resize, pad utility
        self.cropresize = CropResizePad(224)

        # Initialize ROS node and action server
        rospy.init_node("sam6dism_segmentation")
        self.server = actionlib.SimpleActionServer('/object_detector/sam6dism',
                                                   GenericImgProcAnnotatorAction,
                                                   execute_cb=self.segment_objects,
                                                   auto_start=False)
        self.server.start()
        print("Instance Segmentation with SAM-6D ISM is ready.")

    def _initialize_model(self):
        """Initialize the SAM-6D ISM model"""
        print("=> creating ISM model ...")
        
        # Initialize Hydra configuration
        with initialize(version_base=None, config_path="../Instance_Segmentation_Model/configs"):
            cfg = compose(config_name="run_inference.yaml")
            
        # Override model config
        cfg.model = self.model_name
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        model = instantiate(cfg.model)
        model = model.to(self.device)
        model.eval()
        
        self.model = model
        self.cfg = cfg
        
        print("=> ISM model initialization complete")

    def _preprocess_image(self, image):
        """Preprocess image for ISM model"""
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Apply preprocessing
        rgb_tensor = self.rgb_transform(image).unsqueeze(0).to(self.device)
        return rgb_tensor, image

    def _run_detection(self, rgb_tensor, original_image):
        """Run object detection and segmentation"""
        print("=> running ISM inference...")
        
        with torch.no_grad():
            # Run model inference
            detections = self.model(rgb_tensor, run_inference=True)
            
        # Process detections
        if isinstance(detections, list) and len(detections) > 0:
            detections = detections[0]  # Take first batch item
            
        return detections

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
            from segment_anything.utils.amg import mask_to_rle_pytorch
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

    def segment_objects(self, req):
        print("Request segmentation...")
        start_time = time.time()

        # Extract data from request
        rgb = req.rgb
        # depth is optional for ISM
        
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
        
        try:
            # Preprocess image
            rgb_tensor, original_image = self._preprocess_image(image)
            
            # Run detection
            detections = self._run_detection(rgb_tensor, original_image)
            
            # Convert to COCO format
            results = self._convert_to_coco_format(detections, width, height)
            
            # Save results to JSON file
            output_file = os.path.join(self.output_dir, "detection_ism.json")
            with open(output_file, 'w') as f:
                json.dump(results, f)
            
            print(f"=> saved {len(results)} detections to {output_file}")
            
            # Convert detections to ROS format for response
            response_masks = []
            response_class_names = []
            
            for detection in results:
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

        # Create response
        response = GenericImgProcAnnotatorResult()
        response.mask_detections = response_masks
        response.class_names = response_class_names
        response.pose_results = []  # ISM doesn't provide poses

        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Execution time:', elapsed_time, 'seconds')
        
        self.server.set_succeeded(response)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', 
                       default="./configs/cfg_ros_ycbv_inference.json",
                       help='Path to configuration file')
    parser.add_argument('--output_dir',
                       default="/tmp/sam6d_ism",
                       help='Directory to save ISM results')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    sam6dism_ros = SAM6DISM_ROS(**vars(opt))
    
    rospy.spin()
