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

# Add current directory to path for imports - Docker environment
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)
sys.path.append(os.path.join(code_dir, 'provider'))
sys.path.append(os.path.join(code_dir, 'utils'))
sys.path.append(os.path.join(code_dir, 'model'))
sys.path.append(os.path.join(code_dir, 'model', 'pointnet2'))
sys.path.append(os.path.join(code_dir, '..', 'Instance_Segmentation_Model', 'segment_anything'))

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
        self.depth_scale = config.get("depth_scale", 1000.0)
        
        print(f"Using intrinsics: {self.intrinsics}")
        print(f"ISM results directory: {self.ism_results_dir}")

        # Configuration parameters for SAM-6D PEM
        self.det_score_thresh = 0.2
        self.gpus = "0"
        self.model_name = "pose_estimation_model"
        self.config_path = os.path.join(code_dir, "config", "base.yaml")
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
        
        # Setup templates directory base path
        self.templates_base_dir = "/code/datasets/ycbv/templates"
        
        # Initialize all object templates and mappings
        self.object_templates = {}
        self.object_id_to_name = {}
        self.object_name_to_id = {}
        
        for idx, (object_name, object_id) in enumerate(self.object_name_mapping.items()):
            self.object_id_to_name[idx] = object_name
            self.object_name_to_id[object_name] = idx
            
            # Store template directory for each object
            template_dir = f"{self.templates_base_dir}/{object_id}"
            self.object_templates[object_name] = {
                'template_dir': template_dir,
                'object_id': object_id,
                'mesh_path': self.mesh_files.get(object_name)
            }
        
        print(f"Initialized {len(self.object_templates)} objects:")
        for name, info in self.object_templates.items():
            print(f"  {name} -> {info['object_id']} (templates: {info['template_dir']})")

        self.meshes = {}  # Store loaded meshes
        self.model_points = {}  # Store sampled model points for each object
        self.templates = {}  # Store templates for each object
        
        # RGB transform for preprocessing
        self.rgb_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load configuration for test dataset
        self._load_config()
        
        # Pre-load meshes and prepare model points
        self._load_meshes()

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
        for obj_name, obj_info in self.object_templates.items():
            mesh_file = obj_info['mesh_path']
            if mesh_file and os.path.exists(mesh_file):
                mesh = trimesh.load(mesh_file)
                self.meshes[obj_name] = mesh
                
                # Sample model points (convert mm to meters like original SAM-6D)
                model_points = mesh.sample(self.test_config['n_sample_model_point']).astype(np.float32) / 1000.0
                self.model_points[obj_name] = model_points
                
                rospy.loginfo(f"Loaded mesh for {obj_name}: {mesh_file}")
            else:
                rospy.logwarn(f"Mesh file not found for {obj_name}: {mesh_file}")

    def _get_template(self, template_path, tem_index=1):
        """Load a single template view - following original template approach"""
        rgb_path = os.path.join(template_path, f'rgb_{tem_index}.png')
        mask_path = os.path.join(template_path, f'mask_{tem_index}.png')
        xyz_path = os.path.join(template_path, f'xyz_{tem_index}.npy')

        if not all(os.path.exists(p) for p in [rgb_path, mask_path, xyz_path]):
            return None, None, None

        rgb = load_im(rgb_path).astype(np.uint8)
        xyz = np.load(xyz_path).astype(np.float32) / 1000.0  # Convert mm to meters
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
        """Load all template views for an object - following original approach"""
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

    def _extract_template_features(self, obj_name):
        """Extract template features for an object - following original approach"""
        if obj_name in self.templates:
            return self.templates[obj_name]
            
        print(f"=> extracting templates for {obj_name}...")
        
        # Get template directory for this object
        if obj_name not in self.object_templates:
            rospy.logwarn(f"No template info found for {obj_name}")
            return None, None
            
        template_path = self.object_templates[obj_name]['template_dir']
        
        if not os.path.exists(template_path):
            rospy.logwarn(f"Template directory not found: {template_path}")
            return None, None
        
        all_tem, all_tem_pts, all_tem_choose = self._get_templates(template_path)
        
        if all_tem is None:
            rospy.logwarn(f"No templates found for {obj_name} at {template_path}")
            return None, None
            
        # Extract features exactly like original code
        with torch.no_grad():
            all_tem_pts_feat, all_tem_feat = self.model.feature_extraction.get_obj_feats(
                all_tem, all_tem_pts, all_tem_choose)
        
        # Store templates
        self.templates[obj_name] = (all_tem_pts_feat, all_tem_feat)
        return all_tem_pts_feat, all_tem_feat

    def _process_segmentation(self, mask, rgb_img, depth_img, obj_name):
        """Process segmentation mask and extract features - following original get_test_data logic"""
        # Get whole image point cloud
        whole_pts = get_point_cloud_from_depth(depth_img, self.intrinsics)
        
        # Get model points for this object
        model_points = self.model_points[obj_name]
        radius = np.max(np.linalg.norm(model_points, axis=1))

        # Process mask - similar to original code
        mask = np.logical_and(mask > 0, depth_img > 0)
        if np.sum(mask) <= 32:
            return None
            
        bbox = get_bbox(mask)
        y1, y2, x1, x2 = bbox
        mask = mask[y1:y2, x1:x2]
        choose = mask.astype(np.float32).flatten().nonzero()[0]

        # Process point cloud - exactly like original
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

        # Process RGB - exactly like original
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

    def _load_ism_results_for_objects(self, class_names):
        """Load ISM results from JSON files for specific objects"""
        all_detections = {}
        
        for class_name in class_names:
            # Look for JSON file with object name
            json_file = os.path.join(self.ism_results_dir, f"{class_name}.json")
            
            if not os.path.exists(json_file):
                print(f"ISM results file not found for {class_name}: {json_file}")
                continue
                
            try:
                with open(json_file, 'r') as f:
                    detections = json.load(f)
                
                # Use all detections without confidence threshold filtering
                if detections:
                    all_detections[class_name] = detections
                    print(f"=> loaded {len(detections)} detections for {class_name}")
                else:
                    print(f"=> no detections found for {class_name}")
                    
            except Exception as e:
                print(f"Failed to load ISM results for {class_name}: {str(e)}")
                
        return all_detections

    def _convert_rle_to_mask(self, rle_data, image_shape):
        """Convert RLE segmentation to mask - similar to original get_test_data function"""
        try:
            h, w = image_shape[:2]
            # Use the same approach as in the original get_test_data function
            try:
                rle = cocomask.frPyObjects(rle_data, h, w)
            except:
                rle = rle_data
            mask = cocomask.decode(rle)
            return mask
        except Exception as e:
            print(f"Error converting RLE to mask: {e}")
            return None



    def estimate_pose(self, req):
        print("Request pose estimation...")
        start_time = time.time()

        # Extract data from request
        class_names = req.class_names
        rgb = req.rgb
        depth = req.depth
        print(class_names)
        print(f"Received class names: {class_names}")
        print(f"Number of objects to estimate pose for: {len(class_names)}")
        for i, name in enumerate(class_names):
            print(f"  Object {i+1}: {name}")

        width, height = rgb.width, rgb.height
        print(f"Image dimensions: {width}x{height}")

        # Convert ROS messages to numpy arrays
        image = ros_numpy.numpify(rgb)
        if len(image.shape) == 3 and image.shape[2] == 3:
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
        else:
            print(f"WARNING: Unexpected RGB image format: {image.shape}, dtype: {image.dtype}")

        try:
            depth_img = ros_numpy.numpify(depth)
            if depth_img.dtype == np.uint16:
                depth_img = depth_img.astype(np.float32) / self.depth_scale
            elif depth_img.dtype != np.float32:
                depth_img = depth_img.astype(np.float32)
            print("Depth image: Available")
            print(f"Depth range: min={depth_img.min():.3f}, max={depth_img.max():.3f}")
        except Exception as e:
            rospy.logwarn("Missing depth image in the goal.")
            depth_img = None
            print("Depth image: Not available")

        if depth_img is None:
            print("ERROR: Depth image is required for pose estimation")
            response = GenericImgProcAnnotatorResult()
            response.pose_results = []
            response.class_names = []
            self.server.set_aborted()
            return

        print("RGB", image.shape, image.dtype)
        print("Depth", depth_img.shape, depth_img.dtype)

        ism_detections = self._load_ism_results_for_objects(class_names)
        if not ism_detections:
            print("No ISM detections found for any requested objects")
            response = GenericImgProcAnnotatorResult()
            response.pose_results = []
            response.class_names = []
            self.server.set_succeeded(response)
            return

        valid_class_names = []
        pose_results = []
        batch_data = []
        batch_objects = []

        for class_name, detections in ism_detections.items():
            print(f"Processing {len(detections)} detections for {class_name}")
            if class_name not in self.object_name_mapping:
                print(f"  -> No mapping found for {class_name}, skipping")
                continue
            mapped_name = self.object_name_mapping[class_name]
            print(f"  -> Mapped to: {mapped_name}")
            mapped_name = class_name  # Bug should be fixed in the future

            if class_name not in self.meshes:
                print(f"  -> No mesh available for {mapped_name}, skipping")
                continue

            # Filter detections by score threshold
            filtered_detections = [d for d in detections if d.get('score', 0.0) >= self.det_score_thresh]
            # Sort by score descending and keep only top 10 to save on VRAM
            filtered_detections = sorted(filtered_detections, key=lambda d: d.get('score', 0.0), reverse=True)[:10]

            if not filtered_detections:
                print(f"  -> No detections above threshold {self.det_score_thresh} for {class_name}")
                continue

            for det_idx, detection in enumerate(filtered_detections):
                score = detection.get('score', 0.0)
                print(f"  -> Processing detection {det_idx + 1}/{len(filtered_detections)} for {class_name} (score: {score:.3f})")
                try:
                    rle_data = detection["segmentation"]
                    mask = self._convert_rle_to_mask(rle_data, (height, width))
                    if mask is None:
                        print(f"    -> Failed to convert RLE to mask")
                        continue
                    mask_pixels = np.sum(mask > 0)
                    print(f"    -> Mask has {mask_pixels} non-zero pixels")
                    if mask_pixels == 0:
                        print(f"    -> Empty mask, skipping")
                        continue

                    processed_data = self._process_segmentation(mask, image, depth_img, mapped_name)
                    if processed_data is None:
                        print(f"    -> Failed to process segmentation")
                        continue

                    template_features = self._extract_template_features(mapped_name)
                    if template_features is None:
                        print(f"    -> Failed to extract template features for {mapped_name}")
                        continue

                    batch_data.append(processed_data)
                    batch_objects.append((class_name, mapped_name, template_features, score))
                except Exception as e:
                    print(f"    -> Error processing detection: {str(e)}")
                    continue

        if not batch_data:
            print("No valid detections to process")
            response = GenericImgProcAnnotatorResult()
            response.pose_results = []
            response.class_names = []
            self.server.set_succeeded(response)
            return

        print(f"Processing {len(batch_data)} valid detections")

        all_rgb = torch.stack([data['rgb'] for data in batch_data]).cuda()
        all_cloud = torch.stack([data['cloud'] for data in batch_data]).cuda()
        all_rgb_choose = torch.stack([data['rgb_choose'] for data in batch_data]).cuda()
        ninstance = all_rgb.size(0)

        results_by_batch = []
        object_groups = {}
        for idx, (class_name, mapped_name, template_features, score) in enumerate(batch_objects):
            if mapped_name not in object_groups:
                object_groups[mapped_name] = []
            object_groups[mapped_name].append((idx, class_name, template_features, score))

        print("=> running model inference...")

        for mapped_name, group_info in object_groups.items():
            indices = [item[0] for item in group_info]
            class_names_group = [item[1] for item in group_info]
            template_features = group_info[0][2]
            print(f"  -> Processing {len(indices)} instances of {mapped_name}")

            group_rgb = all_rgb[indices]
            group_cloud = all_cloud[indices]
            group_rgb_choose = all_rgb_choose[indices]
            group_size = len(indices)
            model_points = self.model_points[mapped_name]
            group_model = torch.FloatTensor(model_points).unsqueeze(0).repeat(group_size, 1, 1).cuda()
            group_K = torch.FloatTensor(self.intrinsics).unsqueeze(0).repeat(group_size, 1, 1).cuda()

            input_data = {
                'pts': group_cloud,
                'rgb': group_rgb,
                'rgb_choose': group_rgb_choose,
                'model': group_model,
                'K': group_K
            }

            try:
                with torch.no_grad():
                    all_tem_pts, all_tem_feat = template_features
                    input_data['dense_po'] = all_tem_pts.repeat(group_size, 1, 1)
                    input_data['dense_fo'] = all_tem_feat.repeat(group_size, 1, 1)
                    out = self.model(input_data)

                if 'pred_pose_score' in out.keys():
                    pose_scores = out['pred_pose_score'].detach().cpu().numpy()
                else:
                    pose_scores = np.ones(group_size)
                # If out has 'score', use it as ISM detection score multiplier
                if 'score' in out.keys():
                    detection_scores = out['score'].detach().cpu().numpy()
                else:
                    # fallback: use ISM score from group_info
                    detection_scores = np.array([item[3] for item in group_info])

                final_scores = pose_scores * detection_scores
                pred_rot = out['pred_R'].detach().cpu().numpy()
                pred_trans = out['pred_t'].detach().cpu().numpy()

                for i, (orig_idx, class_name, _, ism_score) in enumerate(group_info):
                    results_by_batch.append({
                        'class_name': class_name,
                        'rotation': pred_rot[i],
                        'translation': pred_trans[i],
                        'pose_score': final_scores[i],
                        'ism_score': ism_score,
                        'original_index': orig_idx,
                        'mapped_name': mapped_name
                    })
            except Exception as e:
                print(f"  -> Model inference failed for {mapped_name}: {str(e)}")
                continue

        print("=> processing results...")

        # Print all object instances with pose and ISM score
        print("All detected object instances with pose and ISM score:")
        for result in results_by_batch:
            class_name = result['class_name']
            translation = result['translation']
            pose_score = result['pose_score']
            ism_score = result['ism_score']
            print(f"  {class_name}: Position [{translation[0]:.6f}, {translation[1]:.6f}, {translation[2]:.6f}], "
              f"Pose Score: {pose_score:.6f}, ISM Score: {ism_score:.6f}")

        # Only keep the best pose per object (highest pose_score)
        best_results = {}
        for result in results_by_batch:
            key = result['class_name']
            if key not in best_results or result['pose_score'] > best_results[key]['pose_score']:
                best_results[key] = result

        # Sort results by original index to maintain order if needed
        sorted_best_results = sorted(best_results.values(), key=lambda x: x['original_index'])
        #sorted_best_results = best_results

        for result in sorted_best_results:
            try:
                class_name = result['class_name']
                rotation_matrix = result['rotation']
                translation = result['translation']
                pose_score = result['pose_score']
                ism_score = result['ism_score']

                if np.allclose(translation, 0) and np.allclose(rotation_matrix, np.eye(3)):
                    print(f"  -> WARNING: Invalid pose for {class_name}")
                    continue

                pose_msg = Pose()
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
                print(f"     Pose Score: {pose_score:.6f}, ISM Score: {ism_score:.6f}")

                pose_results.append(pose_msg)
                valid_class_names.append(class_name)
            except Exception as e:
                print(f"  -> Error processing result for {result['class_name']}: {str(e)}")

        response = GenericImgProcAnnotatorResult()
        response.pose_results = pose_results
        response.class_names = valid_class_names

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f'Total execution time: {elapsed_time:.2f} seconds')
        print(f'Successfully estimated poses for {len(pose_results)} object instances')

        self.server.set_succeeded(response)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', 
                       default="/code/configs/cfg_ros_ycbv_inference.json",
                       help='Path to configuration file')
    parser.add_argument('--ism_results_dir',
                       default="/code/tmp",
                       help='Directory where ISM results are saved')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    sam6dpem_ros = SAM6DPEM_ROS(**vars(opt))
    
    rospy.spin()
