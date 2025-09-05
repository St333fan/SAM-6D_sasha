import blenderproc as bproc
import os
import argparse
import cv2
import numpy as np
import trimesh

parser = argparse.ArgumentParser()
parser.add_argument('--cad_path', help="The path of CAD model")
parser.add_argument('--output_dir', help="The path to save CAD templates")
parser.add_argument('--normalize', default=True, help="Whether to normalize CAD model or not")
parser.add_argument('--colorize', default=False, help="Whether to colorize CAD model or not")
parser.add_argument('--base_color', default=0.05, help="The base color used in CAD model")
args = parser.parse_args()

# set the cnos camera path
render_dir = os.path.dirname(os.path.abspath(__file__))
cnos_cam_fpath = os.path.join(render_dir, '../Instance_Segmentation_Model/utils/poses/predefined_poses/cam_poses_level0.npy')

def convert_ply_to_obj(ply_path):
    """Convert PLY file to OBJ format for better compatibility"""
    try:
        # Load the PLY file
        mesh = trimesh.load(ply_path, force='mesh')
        
        # Clean the mesh
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces() 
        mesh.remove_unreferenced_vertices()
        
        # Apply rotation correction during conversion (90 degrees right around Z-axis)

        rotation_matrix = trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0])


        mesh.apply_transform(rotation_matrix)
        print("Applied 90° clockwise rotation around Z-axis during PLY->OBJ conversion")
        
        # Ensure triangulation
        if hasattr(mesh, 'faces') and len(mesh.faces) > 0:
            # Check if we need triangulation
            if mesh.faces.shape[1] > 3:
                mesh = mesh.triangulate()
        
        # Create output path
        obj_path = ply_path.replace('.ply', '_converted.obj')
        
        # Export as OBJ
        mesh.export(obj_path)
        
        print(f"Successfully converted PLY to OBJ: {obj_path}")
        return obj_path
        
    except Exception as e:
        print(f"Error converting PLY file: {e}")
        return ply_path

def get_norm_info(mesh_path):
    mesh = trimesh.load(mesh_path, force='mesh')
    model_points = trimesh.sample.sample_surface(mesh, 1024)[0]
    model_points = model_points.astype(np.float32)
    min_value = np.min(model_points, axis=0)
    max_value = np.max(model_points, axis=0)
    radius = max(np.linalg.norm(max_value), np.linalg.norm(min_value))
    return 1/(2*radius)

bproc.init()

# Convert PLY to OBJ if necessary
if args.cad_path.endswith('.ply'):
    print(f"PLY file detected, converting to OBJ format...")
    args.cad_path = convert_ply_to_obj(args.cad_path)

# load cnos camera pose
cam_poses = np.load(cnos_cam_fpath)

# calculating the scale of CAD model
if args.normalize:
    scale = get_norm_info(args.cad_path)
else:
    scale = 1

for idx, cam_pose in enumerate(cam_poses):
    
    bproc.clean_up()
    
    try:
        # load object
        obj = bproc.loader.load_obj(args.cad_path)[0]
        obj.set_scale([scale, scale, scale])
        obj.set_cp("category_id", 1)
        
        # assigning material colors to untextured objects
        if args.colorize:
            color = [args.base_color, args.base_color, args.base_color, 0.]
            material = bproc.material.create('obj')
            material.set_principled_shader_value('Base Color', color)
            obj.set_material(0, material)
            
        # convert cnos camera poses to blender camera poses
        cam_pose[:3, 1:3] = -cam_pose[:3, 1:3]
        cam_pose[:3, -1] = cam_pose[:3, -1] * 0.001 * 2
        bproc.camera.add_camera_pose(cam_pose)
        
        # set light
        light_scale = 2.5
        light_energy = 1000
        light1 = bproc.types.Light()
        light1.set_type("POINT")
        light1.set_location([light_scale*cam_pose[:3, -1][0], light_scale*cam_pose[:3, -1][1], light_scale*cam_pose[:3, -1][2]])
        light1.set_energy(light_energy)
        bproc.renderer.set_max_amount_of_samples(50)
        
        # render the whole pipeline
        data = bproc.renderer.render()
        # render nocs
        data.update(bproc.renderer.render_nocs())
        
        # check save folder
        save_fpath = os.path.join(args.output_dir, "templates")
        if not os.path.exists(save_fpath):
            os.makedirs(save_fpath)
            
        # save rgb image
        color_bgr_0 = data["colors"][0]
        color_bgr_0[..., :3] = color_bgr_0[..., :3][..., ::-1]
        cv2.imwrite(os.path.join(save_fpath,'rgb_'+str(idx)+'.png'), color_bgr_0)
        
        # save mask
        mask_0 = data["nocs"][0][..., -1]
        cv2.imwrite(os.path.join(save_fpath,'mask_'+str(idx)+'.png'), mask_0*255)
        
        # save nocs
        xyz_0 = 2*(data["nocs"][0][..., :3] - 0.5)
        np.save(os.path.join(save_fpath,'xyz_'+str(idx)+'.npy'), xyz_0.astype(np.float16))
        
        print(f"Rendered template {idx+1}/{len(cam_poses)}")
        
    except Exception as e:
        print(f"Error processing camera pose {idx}: {e}")
        continue

print("Template rendering completed!")

# Clean up converted file if it was created
if args.cad_path.endswith('_converted.obj'):
    try:
        os.remove(args.cad_path)
        print("Cleaned up temporary OBJ file")
    except:
        pass
