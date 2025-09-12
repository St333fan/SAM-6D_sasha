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
import trimesh
import glob
from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torchvision.transforms as T
import imageio
import distinctipy
from skimage.feature import canny
from skimage.morphology import binary_dilation

# Add current directory to path for imports
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)
sys.path.append(os.path.join(code_dir, 'Instance_Segmentation_Model'))
sys.path.append(os.path.join(code_dir, 'Instance_Segmentation_Model', 'model'))
sys.path.append(os.path.join(code_dir, 'Instance_Segmentation_Model', 'utils'))
sys.path.append(os.path.join(code_dir, 'Instance_Segmentation_Model', 'provider'))
sys.path.append(os.path.join(code_dir, 'Instance_Segmentation_Model', 'segment_anything'))

from model.utils import Detections, convert_npz_to_json
from model.loss import Similarity
from utils.bbox_utils import CropResizePad
from utils.poses.pose_utils import get_obj_poses_from_template_level, load_index_level_in_level2
from utils.inout import load_json, save_json_bop23
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
        self.segmentor_model = config.get("segmentor_model", "fastsam")
        self.stability_score_thresh = config.get("stability_score_thresh", 0.97)
        self.depth_scale = config.get("depth_scale", 1000.0)
        
        print(f"Using intrinsics: {self.intrinsics}")
        print(f"Output directory: {self.output_dir}")
        print(f"Segmentor model: {self.segmentor_model}")

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}", exist_ok=True)

        # Store mesh information for each object
        self.mesh_files = config.get('mesh_files', {})
        
        # Setup paths for models and templates
        self.models_dir = "code/datasets/ycbv/models"
        self.templates_dir = "code/datasets/ycbv/templates"

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
        # Initialize the model
        self._initialize_model()

        # Initialize templates
        self._initialize_templates(self.templates_dir)
        
        # Initialize ROS node and action server
        rospy.init_node("sam6dism_segmentation")
        self.server = actionlib.SimpleActionServer('/object_detector/sam6dism',
                                                   GenericImgProcAnnotatorAction,
                                                   execute_cb=self.segment_objects,
                                                   auto_start=False)
        self.server.start()
        print("Instance Segmentation with SAM-6D ISM is ready.")

    def _initialize_model(self):
        """Initialize the SAM-6D ISM model with complete pipeline"""
        # Change to Instance_Segmentation_Model directory for relative imports
        original_cwd = os.getcwd()
        ism_dir = os.path.join(code_dir, 'Instance_Segmentation_Model')
        os.chdir(ism_dir)
        
        try:
            with initialize(version_base=None, config_path="configs"):
                cfg = compose(config_name='run_inference.yaml')

            if self.segmentor_model == "sam":
                with initialize(version_base=None, config_path="configs/model"):
                    cfg.model = compose(config_name='ISM_sam.yaml')
                cfg.model.segmentor_model.stability_score_thresh = self.stability_score_thresh
            elif self.segmentor_model == "fastsam":
                with initialize(version_base=None, config_path="configs/model"):
                    cfg.model = compose(config_name='ISM_fastsam.yaml')
            else:
                raise ValueError(f"The segmentor_model {self.segmentor_model} is not supported now!")

            logging.info("Initializing model")
            model = instantiate(cfg.model)
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.descriptor_model.model = model.descriptor_model.model.to(self.device)
            model.descriptor_model.model.device = self.device
            
            # if there is predictor in the model, move it to device
            if hasattr(model.segmentor_model, "predictor"):
                model.segmentor_model.predictor.model = (
                    model.segmentor_model.predictor.model.to(self.device)
                )
            else:
                model.segmentor_model.model.setup_model(device=self.device, verbose=True)
            
            logging.info(f"Moving models to {self.device} done!")
            
            self.model = model
            self.cfg = cfg
            
        finally:
            # Change back to original directory
            os.chdir(original_cwd)
        
        print("=> ISM model initialization complete")

    def _initialize_templates(self, template_dir):
        """Initialize templates for the object detection"""
        num_templates = len(glob.glob(f"{template_dir}/*.npy"))
        if num_templates == 0:
            logging.warning(f"No templates found in {template_dir}")
            return False
            
        boxes, masks, templates = [], [], []
        for idx in range(num_templates):
            rgb_path = os.path.join(template_dir, f'rgb_{idx}.png')
            mask_path = os.path.join(template_dir, f'mask_{idx}.png')
            
            if not os.path.exists(rgb_path) or not os.path.exists(mask_path):
                continue
                
            image = Image.open(rgb_path)
            mask = Image.open(mask_path)
            boxes.append(mask.getbbox())

            image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
            mask = torch.from_numpy(np.array(mask.convert("L")) / 255).float()
            image = image * mask[:, :, None]
            templates.append(image)
            masks.append(mask.unsqueeze(-1))
            
        if len(templates) == 0:
            logging.warning("No valid templates found")
            return False
            
        templates = torch.stack(templates).permute(0, 3, 1, 2)
        masks = torch.stack(masks).permute(0, 3, 1, 2)
        boxes = torch.tensor(np.array(boxes))
        
        processing_config = OmegaConf.create({"image_size": 224})
        proposal_processor = CropResizePad(processing_config.image_size)
        templates = proposal_processor(images=templates, boxes=boxes).to(self.device)
        masks_cropped = proposal_processor(images=masks, boxes=boxes).to(self.device)

        self.model.ref_data = {}
        self.model.ref_data["descriptors"] = self.model.descriptor_model.compute_features(
                        templates, token_name="x_norm_clstoken"
                    ).unsqueeze(0).data
        self.model.ref_data["appe_descriptors"] = self.model.descriptor_model.compute_masked_patch_feature(
                        templates, masks_cropped[:, 0, :, :]
                    ).unsqueeze(0).data
        return True

    def _prepare_batch_data(self, depth_image):
        """Prepare batch data from depth and camera info"""
        batch = {}
        depth = depth_image.astype(np.int32)
        cam_K = self.intrinsics
        depth_scale = np.array(self.depth_scale)

        batch["depth"] = torch.from_numpy(depth).unsqueeze(0).to(self.device)
        batch["cam_intrinsic"] = torch.from_numpy(cam_K).unsqueeze(0).to(self.device)
        batch['depth_scale'] = torch.from_numpy(depth_scale).unsqueeze(0).to(self.device)
        return batch

    def _run_sam6d_inference(self, rgb_image, depth_image=None, cad_path=None):
        """Run complete SAM-6D ISM inference pipeline"""
        # Convert PIL image to numpy if needed
        if isinstance(rgb_image, Image.Image):
            rgb_array = np.array(rgb_image)
        else:
            rgb_array = rgb_image
            rgb_image = Image.fromarray(rgb_array)
        
        # Run segmentation
        detections = self.model.segmentor_model.generate_masks(rgb_array)
        detections = Detections(detections)
        
        if len(detections) == 0:
            logging.warning("No detections found")
            return None
        
        # Compute descriptors
        query_descriptors, query_appe_descriptors = self.model.descriptor_model.forward(rgb_array, detections)

        # Matching descriptors
        (
            idx_selected_proposals,
            pred_idx_objects,
            semantic_score,
            best_template,
        ) = self.model.compute_semantic_score(query_descriptors)

        # Update detections
        detections.filter(idx_selected_proposals)
        query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]

        # Compute appearance score
        appe_scores, ref_aux_descriptor = self.model.compute_appearance_score(
            best_template, pred_idx_objects, query_appe_descriptors)

        # Compute geometric score if depth and CAD are available
        if depth_image is not None and cad_path is not None and os.path.exists(cad_path):
            try:
                batch = self._prepare_batch_data(depth_image)
                template_poses = get_obj_poses_from_template_level(level=2, pose_distribution="all")
                template_poses[:, :3, 3] *= 0.4
                poses = torch.tensor(template_poses).to(torch.float32).to(self.device)
                self.model.ref_data["poses"] = poses[load_index_level_in_level2(0, "all"), :, :]

                mesh = trimesh.load_mesh(cad_path)
                model_points = mesh.sample(2048).astype(np.float32) / 1000.0
                self.model.ref_data["pointcloud"] = torch.tensor(model_points).unsqueeze(0).data.to(self.device)
                
                image_uv = self.model.project_template_to_image(
                    best_template, pred_idx_objects, batch, detections.masks)

                geometric_score, visible_ratio = self.model.compute_geometric_score(
                    image_uv, detections, query_appe_descriptors, ref_aux_descriptor, 
                    visible_thred=self.model.visible_thred)

                # Final score with geometric component
                final_score = (semantic_score + appe_scores + geometric_score*visible_ratio) / (1 + 1 + visible_ratio)
            except Exception as e:
                logging.warning(f"Geometric scoring failed: {e}, using semantic + appearance only")
                final_score = (semantic_score + appe_scores) / 2
        else:
            # Final score without geometric component
            final_score = (semantic_score + appe_scores) / 2
            
        detections.add_attribute("scores", final_score)
        detections.add_attribute("object_ids", torch.zeros_like(final_score))   
             
        detections.to_numpy()
        
        # Save results as JSON only (no images as requested)
        save_path = f"{self.output_dir}"
        detections.save_to_file(0, 0, 0, save_path, "Custom", return_results=False)
        detections_json = convert_npz_to_json(idx=0, list_npz_paths=[save_path+".npz"])
        save_json_bop23(save_path+".json", detections_json)
        
        return detections_json

    def segment_objects(self, req):
        print("Request segmentation...")
        start_time = time.time()

        # Extract data from request
        rgb = req.rgb
        depth = getattr(req, 'depth', None)  # depth is optional for ISM
        
        # Get image dimensions
        width, height = rgb.width, rgb.height
        print(f"Image dimensions: {width}x{height}")
        
        # Convert ROS message to numpy array
        rgb_image = ros_numpy.numpify(rgb)
        depth_image = ros_numpy.numpify(depth) if depth is not None else None
        
        # For demonstration, we'll assume templates are in the output directory
        # In practice, you might want to specify template directories per object
        template_dir = os.path.join(self.output_dir, 'templates')
        
        # For demonstration, we'll use the first available CAD file
        cad_path = None
        if self.mesh_files:
            cad_path = list(self.mesh_files.values())[0]
        
        # Run SAM-6D ISM inference
        try:
            detections_result = self._run_sam6d_inference(
                rgb_image, 
                depth_image=depth_image, 
                cad_path=cad_path,
                template_dir=template_dir if os.path.exists(template_dir) else None
            )
            
            if detections_result is None:
                print("No detections found")
                response_masks = []
                response_class_names = []
            else:
                # Convert detections to ROS format
                response_masks = []
                response_class_names = []
                
                for detection in detections_result:
                    # Create mask message
                    mask_msg = ROSImage()
                    if 'segmentation' in detection:
                        # Convert RLE to mask
                        mask = rle_to_mask(detection["segmentation"])
                        mask_img = (mask * 255).astype(np.uint8)
                        mask_msg = ros_numpy.msgify(ROSImage, mask_img, encoding='mono8')
                        response_masks.append(mask_msg)
                        
                        # Add class name (for now, using generic name)
                        class_name = f"object_{detection.get('category_id', 0)}"
                        response_class_names.append(class_name)
                
                print(f"Found {len(response_masks)} detections")
                
        except Exception as e:
            print(f"Error in SAM-6D ISM inference: {e}")
            import traceback
            traceback.print_exc()
            response_masks = []
            response_class_names = []

        # Create response
        response = GenericImgProcAnnotatorResult()
        response.mask_detections = response_masks
        response.class_names = response_class_names

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
