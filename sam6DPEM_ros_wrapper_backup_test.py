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

# Add current directory to path for imports
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)
sys.path.append(os.path.join(code_dir, 'Pose_Estimation_Model'))
sys.path.append(os.path.join(code_dir, 'Pose_Estimation_Model', 'provider'))
sys.path.append(os.path.join(code_dir, 'Pose_Estimation_Model', 'utils'))
sys.path.append(os.path.join(code_dir, 'Pose_Estimation_Model', 'model'))
sys.path.append(os.path.join(code_dir, 'Pose_Estimation_Model', 'model', 'pointnet2'))

import torch
import torchvision.transforms as transforms
import gorilla
import pycocotools.mask as cocomask

# Import custom modules
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

class SAM6DPEM_ROS:
    def __init__(self, config_file, ism_results_dir="/tmp/sam6d_ism"):
        print(f"Using config file: {config_file}")
        with open(config_file, 'r') as f:
            config = json.load(f)
        self.object_name_mapping = config["object_mapping"]
        self.intrinsics = np.asarray(config['cam_K']).reshape((3, 3))
        self.ism_results_dir = ism_results_dir
        print(f"Using intrinsics: {self.intrinsics}")
        print(f"ISM results directory: {self.ism_results_dir}")

        # Configuration parameters for SAM-6D PEM
        self.det_score_thresh = 0.4
        self.gpus = "0"
        self.model_name = "pose_estimation_model"
        self.config_path = os.path.join(code_dir, "Pose_Estimation_Model", "config", "base.yaml")
        self.iter = 600000
        self.exp_id = 0

        # Set random seeds
        random.seed(42)
        torch.manual_seed(42)

        rospy.loginfo("Initializing SAM-6D PEM components")
        
        # Set CUDA device
        gorilla.utils.set_cuda_visible_devices(gpu_ids=self.gpus)
        
        # Initialize SAM-6D PEM model
        self._initialize_model()
        
        # Store mesh information for each object
        self.mesh_files = config.get('mesh_files', {})
        self.meshes = {}  # Store loaded meshes
        self.model_points = {}  # Store sampled model points for each object
        self.templates = {}  # Store templates for each object
        
        # RGB transform for preprocessing
        self.rgb_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Pre-load meshes and prepare model points
        self._load_meshes()
        
        # Load configuration for test dataset
        self._load_config()

        # Initialize ROS node and action server
        rospy.init_node("sam6dpem_estimation")
        self.server = actionlib.SimpleActionServer('/pose_estimator/sam6dpem',
                                                   GenericImgProcAnnotatorAction,
                                                   execute_cb=self.estimate_pose,
                                                   auto_start=False)
        self.server.start()
        print("Pose Estimation with SAM-6D PEM is ready.")

    def _initialize_model(self):
        """Initialize the SAM-6D PEM model"""
        print("=> creating model ...")
        
        # Load configuration
        cfg = gorilla.Config.fromfile(self.config_path)
        exp_name = self.model_name + '_' + \
            os.path.splitext(self.config_path.split("/")[-1])[0] + '_id' + str(self.exp_id)
        log_dir = os.path.join("log", exp_name)

        cfg.exp_name = exp_name
        cfg.gpus = self.gpus
        cfg.model_name = self.model_name
        cfg.log_dir = log_dir
        cfg.test_iter = self.iter

        # Import and create model
        MODEL = importlib.import_module(cfg.model_name)
        self.model = MODEL.Net(cfg.model)
        self.model = self.model.cuda()
        self.model.eval()
        
        # Load checkpoint
        checkpoint_path = os.path.join(code_dir, 'Pose_Estimation_Model', 'checkpoints', 'sam-6d-pem-base.pth')
        if os.path.exists(checkpoint_path):
            gorilla.solver.load_checkpoint(model=self.model, filename=checkpoint_path)
            print(f"=> loaded checkpoint: {checkpoint_path}")
        else:
            rospy.logwarn(f"Checkpoint not found: {checkpoint_path}")
            
        self.cfg = cfg
        print("=> model initialization complete")

    def _load_config(self):
        """Load test dataset configuration"""
        # Default configuration values
        self.test_config = {
            'img_size': 224,
            'n_sample_template_point': 2048,
            'n_sample_observed_point': 1024,
            'n_sample_model_point': 5000,
            'n_template_view': 42,
            'rgb_mask_flag': True
        }
        
        # Try to load from model config if available
        if hasattr(self.cfg, 'test_dataset'):
            for key in self.test_config.keys():
                if hasattr(self.cfg.test_dataset, key):
                    self.test_config[key] = getattr(self.cfg.test_dataset, key)

    def _load_meshes(self):
        """Pre-load meshes and sample model points"""
        for obj_name, mesh_file in self.mesh_files.items():
            if os.path.exists(mesh_file):
                mesh = trimesh.load(mesh_file)
                self.meshes[obj_name] = mesh
                
                # Sample model points (convert mm to meters like original SAM-6D)
                model_points = mesh.sample(self.test_config['n_sample_model_point']).astype(np.float32) / 1000.0
                self.model_points[obj_name] = model_points
                
                rospy.loginfo(f"Loaded mesh for {obj_name}: {mesh_file}")
            else:
                rospy.logwarn(f"Mesh file not found for {obj_name}: {mesh_file}")

    def _get_template(self, template_path, tem_index=1):
        """Load a single template view"""
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
        if self.test_config['rgb_mask_flag']:
            rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)

        rgb = cv2.resize(rgb, (self.test_config['img_size'], self.test_config['img_size']), 
                        interpolation=cv2.INTER_LINEAR)
        rgb = self.rgb_transform(np.array(rgb))

        choose = (mask>0).astype(np.float32).flatten().nonzero()[0]
        if len(choose) <= self.test_config['n_sample_template_point']:
            choose_idx = np.random.choice(np.arange(len(choose)), self.test_config['n_sample_template_point'])
        else:
            choose_idx = np.random.choice(np.arange(len(choose)), self.test_config['n_sample_template_point'], replace=False)
        choose = choose[choose_idx]
        xyz = xyz[y1:y2, x1:x2, :].reshape((-1, 3))[choose, :]

        rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], self.test_config['img_size'])
        return rgb, rgb_choose, xyz

    def _get_templates(self, template_path):
        """Load all template views for an object"""
        n_template_view = self.test_config['n_template_view']
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
        """Extract template features for an object"""
        if obj_name in self.templates:
            return self.templates[obj_name]
            
        print(f"=> extracting templates for {obj_name}...")
        all_tem, all_tem_pts, all_tem_choose = self._get_templates(template_path)
        
        if all_tem is None:
            rospy.logwarn(f"No templates found for {obj_name} at {template_path}")
            return None, None
            
        with torch.no_grad():
            all_tem_pts_feat, all_tem_feat = self.model.feature_extraction.get_obj_feats(
                all_tem, all_tem_pts, all_tem_choose)
        
        # Store templates
        self.templates[obj_name] = (all_tem_pts_feat, all_tem_feat)
        return all_tem_pts_feat, all_tem_feat

    def _process_segmentation(self, mask, rgb_img, depth_img, obj_name):
        """Process segmentation mask and extract features"""
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

        if len(choose) <= self.test_config['n_sample_observed_point']:
            choose_idx = np.random.choice(np.arange(len(choose)), self.test_config['n_sample_observed_point'])
        else:
            choose_idx = np.random.choice(np.arange(len(choose)), self.test_config['n_sample_observed_point'], replace=False)
        choose = choose[choose_idx]
        cloud = cloud[choose_idx]

        # Process RGB
        rgb = rgb_img.copy()[y1:y2, x1:x2, :][:,:,::-1]
        if self.test_config['rgb_mask_flag']:
            rgb = rgb * (mask[:,:,None]>0).astype(np.uint8)
        rgb = cv2.resize(rgb, (self.test_config['img_size'], self.test_config['img_size']), 
                        interpolation=cv2.INTER_LINEAR)
        rgb = self.rgb_transform(np.array(rgb))
        rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], self.test_config['img_size'])

        return {
            'rgb': torch.FloatTensor(rgb),
            'cloud': torch.FloatTensor(cloud),
            'rgb_choose': torch.IntTensor(rgb_choose).long(),
            'model_points': model_points
        }

    def _load_ism_results(self, timestamp_or_latest="latest"):
        """Load ISM results from JSON file"""
        ism_file = os.path.join(self.ism_results_dir, "detection_ism.json")
        
        if not os.path.exists(ism_file):
            rospy.logwarn(f"ISM results file not found: {ism_file}")
            return []
            
        try:
            with open(ism_file, 'r') as f:
                detections = json.load(f)
            print(f"=> loaded {len(detections)} detections from ISM")
            return detections
        except Exception as e:
            rospy.logwarn(f"Failed to load ISM results: {str(e)}")
            return []

    def _convert_ism_to_masks(self, ism_detections, image_shape):
        """Convert ISM detections to mask format"""
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
                
                # Map category to class name (you may need to adjust this mapping)
                class_name = f"category_{category_id}"
                for obj_name, mapped_name in self.object_name_mapping.items():
                    # Simple mapping - you might need more sophisticated logic
                    if str(category_id) in str(mapped_name) or obj_name in str(detection):
                        class_name = obj_name
                        break
                
                masks.append(mask)
                class_names.append(class_name)
                scores.append(score)
                
            except Exception as e:
                print(f"  -> Failed to process detection: {str(e)}")
                continue
                
        return masks, class_names, scores

    def estimate_pose(self, req):
        print("Request detection...")
        start_time = time.time()

        # Extract data from request
        mask_detections = req.mask_detections
        class_names = req.class_names
        rgb = req.rgb
        depth = req.depth

        # Print incoming class names for debugging
        print(f"Received class names: {class_names}")
        print(f"Number of detected objects: {len(class_names)}")
        
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

        width, height = rgb.width, rgb.height
        print(f"Image dimensions: {width}x{height}")
        
        # Convert ROS messages to numpy arrays
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
        
        try:
            depth_img = ros_numpy.numpify(depth)
            # Convert depth image to supported dtype (float32) and to meters (like original SAM-6D)
            if depth_img.dtype == np.uint16:
                # Assume depth is in millimeters, convert to meters
                depth_img = depth_img.astype(np.float32) / 1000.0
            elif depth_img.dtype != np.float32:
                depth_img = depth_img.astype(np.float32)
            print("Depth image: Available")
            print(f"Depth range: min={depth_img.min():.3f}, max={depth_img.max():.3f}")
        except Exception as e:
            rospy.logwarn("Missing depth image in the goal.")
            depth_img = None
            print("Depth image: Not available")

        print("RGB", image.shape, image.dtype)
        
        if depth_img is not None:
            print("Depth", depth_img.shape, depth_img.dtype)

        # Convert mask detections
        mask_detections = [ros_numpy.numpify(mask_img).astype(np.uint8)
                          for mask_img in req.mask_detections]

        if mask_detections:
            print("Mask", mask_detections[0].shape, mask_detections[0].dtype)
            print(f"Number of masks: {len(mask_detections)}")
            
            # Ensure masks are binary (0 or 255)
            for i, mask in enumerate(mask_detections):
                if mask.max() <= 1:
                    # Convert from 0-1 to 0-255
                    mask_detections[i] = (mask * 255).astype(np.uint8)

        # Process each detected object
        valid_class_names = []
        pose_results = []

        # Collect all valid detections for batch processing
        batch_data = []
        batch_objects = []
        
        for i, class_name in enumerate(class_names):
            print(f"Processing object {i+1}/{len(class_names)}: {class_name}")
            
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

            if depth_img is None:
                print(f"  -> ERROR: Depth image required for pose estimation but not available")
                continue

            # Process segmentation
            processed_data = self._process_segmentation(mask, image, depth_img, mapped_name)
            if processed_data is None:
                print(f"  -> Failed to process segmentation for {class_name}")
                continue

            # Get templates for this object - extract object ID from mapped_name
            # Convert mapped name like "006_mustard_bottle" to object ID like "obj_000005"
            if mapped_name in self.mesh_files:
                mesh_file = self.mesh_files[mapped_name]
                # Extract object ID from mesh file path (e.g., obj_000005.ply -> obj_000005)
                obj_id = os.path.splitext(os.path.basename(mesh_file))[0]
                template_path = os.path.join("/code", "templates", obj_id)
            else:
                # Fallback to using mapped_name directly
                template_path = os.path.join("/code", "templates", mapped_name)
                
            if not os.path.exists(template_path):
                print(f"  -> Template path not found: {template_path}")
                continue

            template_features = self._extract_template_features(mapped_name, template_path)
            if template_features is None:
                print(f"  -> Failed to extract template features for {mapped_name}")
                continue

            batch_data.append(processed_data)
            batch_objects.append((class_name, mapped_name, template_features))

        if not batch_data:
            print("No valid objects to process")
            response = GenericImgProcAnnotatorResult()
            response.pose_results = []
            response.class_names = []
            self.server.set_succeeded(response)
            return

        # Prepare batch input for model
        all_rgb = torch.stack([data['rgb'] for data in batch_data]).cuda()
        all_cloud = torch.stack([data['cloud'] for data in batch_data]).cuda()
        all_rgb_choose = torch.stack([data['rgb_choose'] for data in batch_data]).cuda()
        
        ninstance = all_rgb.size(0)
        
        # Use the first object's model points and K matrix for batch (assuming same object type)
        model_points = batch_data[0]['model_points']
        all_model = torch.FloatTensor(model_points).unsqueeze(0).repeat(ninstance, 1, 1).cuda()
        all_K = torch.FloatTensor(self.intrinsics).unsqueeze(0).repeat(ninstance, 1, 1).cuda()
        
        input_data = {
            'pts': all_cloud,
            'rgb': all_rgb,
            'rgb_choose': all_rgb_choose,
            'model': all_model,
            'K': all_K
        }

        print("=> running model inference...")
        try:
            with torch.no_grad():
                # Use template features from first object (assuming same type)
                all_tem_pts, all_tem_feat = batch_objects[0][2]
                input_data['dense_po'] = all_tem_pts.repeat(ninstance, 1, 1)
                input_data['dense_fo'] = all_tem_feat.repeat(ninstance, 1, 1)
                
                out = self.model(input_data)

            # Process results
            if 'pred_pose_score' in out.keys():
                pose_scores = out['pred_pose_score']
            else:
                pose_scores = torch.ones(ninstance).cuda()
                
            pose_scores = pose_scores.detach().cpu().numpy()
            pred_rot = out['pred_R'].detach().cpu().numpy()
            pred_trans = out['pred_t'].detach().cpu().numpy() * 1000  # Convert back to mm like original SAM-6D

            print("=> processing results...")
            for idx, (class_name, mapped_name, _) in enumerate(batch_objects):
                try:
                    # Convert pose to ROS Pose message
                    pose_msg = Pose()
                    
                    rotation_matrix = pred_rot[idx]
                    translation = pred_trans[idx]
                    
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
                    print(f"     Score: {pose_scores[idx]:.6f}")
                    
                    pose_results.append(pose_msg)
                    valid_class_names.append(class_name)
                    
                except Exception as e:
                    print(f"  -> Error processing result for {class_name}: {str(e)}")

        except Exception as e:
            print(f"=> Model inference failed: {str(e)}")
            rospy.logwarn(f"Model inference failed: {str(e)}")

        # Create response
        response = GenericImgProcAnnotatorResult()
        response.pose_results = pose_results
        response.class_names = valid_class_names

        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Execution time:', elapsed_time, 'seconds')
        
        self.server.set_succeeded(response)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', 
                       default="./configs/cfg_ros_ycbv_inference.json",
                       help='Path to configuration file')
    parser.add_argument('--ism_results_dir',
                       default="/tmp/sam6d_ism",
                       help='Directory where ISM results are saved')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    sam6dpem_ros = SAM6DPEM_ROS(**vars(opt))
    
    rospy.spin()
