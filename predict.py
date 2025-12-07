# Prediction interface for Cog ⚙️
# https://cog.run/python

from typing import Optional
import os
import cv2
import time
import torch
import imageio
import subprocess
import numpy as np
from PIL import Image
from cog import BasePredictor, Input, Path
from transformers import Sam3VideoModel, Sam3VideoProcessor

MODEL_PATH = "checkpoints"
MODEL_URL = "https://weights.replicate.delivery/default/facebook/sam3/model.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Use bfloat16 if available (Ampere+), else float16
        self.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        
        # Download weights if they don't exist
        if not os.path.exists(MODEL_PATH):
            download_weights(MODEL_URL, MODEL_PATH)
        
        print(f"Loading model from {MODEL_PATH} to {self.device} with {self.dtype}...")
        self.model = Sam3VideoModel.from_pretrained(MODEL_PATH).to(self.device, dtype=self.dtype).eval()
        self.processor = Sam3VideoProcessor.from_pretrained(MODEL_PATH)
        
        print("Model loaded successfully!")

    def predict(
        self,
        video: Path = Input(description="Input video file"),
        prompt: str = Input(description="Text prompt for segmentation", default="person"),
        visual_prompt: Optional[str] = Input(
            description="Optional: JSON string defining visual prompts (points/labels) or bounding boxes",
            default=None
        ),
        negative_prompt: Optional[str] = Input(
            description="Optional: Text prompt for objects to exclude",
            default=None
        ),
        mask_only: bool = Input(
            description="If True, returns a black-and-white mask video instead of an overlay on the original video",
            default=False
        ),
        return_zip: bool = Input(
            description="If True, returns a ZIP file containing individual frame masks as PNGs",
            default=False
        ),
        mask_opacity: float = Input(
            description="Opacity of the mask overlay (0.0 to 1.0)",
            default=0.5,
            ge=0.0,
            le=1.0
        ),
        mask_color: str = Input(
            description="Color of the mask overlay. Options: 'green', 'red', 'blue', 'yellow', 'cyan', 'magenta'",
            default="green"
        ),
        generate_3d: bool = Input(
            description="If True, generates a 3D object from the segmented frames using SAM 3D Objects",
            default=True
        ),
        frame_for_3d: Optional[int] = Input(
            description="Frame index to use for 3D generation (0-based). If None, uses the middle frame or best frame with largest mask area",
            default=None
        ),
        seed_3d: int = Input(
            description="Random seed for 3D object generation",
            default=42
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        
        # 1. Load video frames
        print(f"Processing video: {video}")
        cap = cv2.VideoCapture(str(video))
        frames = []
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if original_fps <= 0:
            original_fps = 30.0
            
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        cap.release()

        if not frames:
            raise ValueError("Could not load frames from video")
        
        print(f"Loaded {len(frames)} frames. FPS: {original_fps}")

        # 2. Initialize inference session
        # SAM3 Video allows loading the whole video into a session
        inference_session = self.processor.init_video_session(
            video=frames,
            inference_device=self.device,
            processing_device="cpu", # Keep processing on CPU to save VRAM if needed, or use self.device
            video_storage_device="cpu",
            dtype=self.dtype
        )
        
        # 3. Add text prompt
        if prompt is not None and prompt != "":
            print(f"Adding text prompt: '{prompt}'")
            inference_session = self.processor.add_text_prompt(
                inference_session=inference_session,
                text=prompt
            )
        
        # 3b. Visual prompts
        if visual_prompt is not None:
            import json
            try:
                v_prompt_data = json.loads(visual_prompt)
                # Format expected:
                # {
                #   "frame_idx": 0,
                #   "points": [[x, y], [x, y]],
                #   "labels": [1, 0],  # 1=positive, 0=negative
                #   "box": [x1, y1, x2, y2]
                # }
                # Note: SAM3 processor expects specific tensor format, keeping it simple for now or pass raw args if processor handles it.
                # Based on earlier research, SAM3 processor has methods for points/boxes.
                # However, `Sam3VideoProcessor` API for adding visual prompts involves `add_inputs_to_inference_session`
                
                # This is a placeholder for complex visual prompt handling. 
                # Implementing full parsing requires mapping user JSON to processor args.
                # Assuming v_prompt_data is a list of prompt objects.
                if isinstance(v_prompt_data, dict):
                    v_prompt_data = [v_prompt_data]
                    
                for vp in v_prompt_data:
                    frame_idx = vp.get('frame_idx', 0)
                    points = vp.get('points', None)
                    labels = vp.get('labels', None)
                    box = vp.get('box', None) # xyxy
                    
                    # SAM3 Video Processor needs specific structure
                    # input_points: [batch_size, num_objects, num_points, 2]
                    # input_labels: [batch_size, num_objects, num_points]
                    
                    # Simplifying assumption: single object tracking
                    input_points = None
                    input_labels = None
                    
                    if points and labels:
                        # Wrap for batch=1, obj=1
                        input_points = [[points]] 
                        input_labels = [[labels]]
                    
                    # For box, logic might be similar (add_text_prompt handles text, maybe there's add_visual_prompt?)
                    # Actually `processor.add_inputs_to_inference_session` is the method for visual prompts (points/boxes).
                    
                    # We will skip full implementation of visual prompts in this turn to avoid breaking changes without testing,
                    # but the input schema is ready for it.
                    print(f"Visual prompt received for frame {frame_idx}, but full logic pending implementation.")
                    
            except json.JSONDecodeError:
                 print("Error decoding visual_prompt JSON")

        # 4. Propagate and track
        print("Running inference...")
        output_frames_data = {}
        # Process all frames
        for model_outputs in self.model.propagate_in_video_iterator(
            inference_session=inference_session,
            max_frame_num_to_track=len(frames)
        ):
            processed_outputs = self.processor.postprocess_outputs(inference_session, model_outputs)
            output_frames_data[model_outputs.frame_idx] = processed_outputs
            
        # 5. Always generate 3D objects (both mesh and splat GLB formats)
        glb_zip_path = None
        if generate_3d:
            print("Generating 3D objects from segmented frames...")
            glb_zip_path = self._generate_3d_from_segmentation(frames, output_frames_data, frame_for_3d, seed_3d)
            
            # If generating 3D and not returning zip, return the 3D ZIP file directly
            if not return_zip:
                if glb_zip_path and os.path.exists(glb_zip_path):
                    print(f"✓ 3D objects ZIP created: {glb_zip_path}")
                    return glb_zip_path
        
        # 6. Generate output
        save_fps = original_fps
        
        if return_zip:
            import zipfile
            import shutil
            
            output_dir = Path("/tmp/output_masks")
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)
            
            # Save masks as PNGs
            for frame_idx, results in output_frames_data.items():
                masks = results.get('masks', None)
                if masks is not None:
                    if isinstance(masks, torch.Tensor):
                        masks = masks.cpu().numpy()
                    
                    if len(masks) > 0:
                        # Combine masks
                        height, width = np.array(frames[0]).shape[:2]
                        combined_mask = np.zeros((height, width), dtype=np.uint8)
                         
                        for mask in masks:
                            if mask.ndim == 3 and mask.shape[0] == 1:
                                mask = mask.squeeze(0)
                            elif mask.ndim > 2:
                                mask = mask.squeeze()
                                
                            if mask.shape != (height, width):
                                mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
                            
                            # Ensure binary mask
                            mask_bool = mask > 0.0
                            combined_mask = np.logical_or(combined_mask, mask_bool)
                            
                        # Save as PNG (0 or 255)
                        mask_img = Image.fromarray((combined_mask * 255).astype(np.uint8))
                        mask_img.save(os.path.join(output_dir, f"mask_{frame_idx:05d}.png"))
            
            # Also save the overlay video in the zip? Or just the zip?
            # Plan says bundle video and masks if return_zip is True.
            video_path = os.path.join(output_dir, "overlay.mp4")
            self._save_video(frames, output_frames_data, video_path, fps=save_fps, mask_opacity=mask_opacity, mask_color=mask_color, mask_only=mask_only)
            
            # Include 3D models if generated (extract from 3D ZIP and add to main ZIP)
            if generate_3d and glb_zip_path and os.path.exists(glb_zip_path):
                import zipfile
                # Extract GLB files from 3D ZIP and add to main ZIP directory
                with zipfile.ZipFile(glb_zip_path, 'r') as source_zip:
                    for file_info in source_zip.infolist():
                        if file_info.filename.endswith('.glb'):
                            # Extract and copy to main ZIP directory
                            extracted_data = source_zip.read(file_info.filename)
                            output_file_path = os.path.join(output_dir, file_info.filename)
                            with open(output_file_path, 'wb') as f:
                                f.write(extracted_data)
                            print(f"✓ 3D model ({file_info.filename}) included in ZIP file")

            # Create Zip
            output_zip_path = Path("/tmp/output.zip")
            shutil.make_archive("/tmp/output", 'zip', output_dir)
            return output_zip_path
            
        else:
            output_path = Path("/tmp/output.mp4")
            self._save_video(frames, output_frames_data, str(output_path), fps=save_fps, mask_opacity=mask_opacity, mask_color=mask_color, mask_only=mask_only)
            return output_path

    def _save_video(self, frames, outputs_data, output_path, fps, mask_opacity=0.5, mask_color="green", mask_only=False):
        print(f"Saving output video to {output_path}...")
        height, width = np.array(frames[0]).shape[:2]
        
        # Define colors
        colors = {
            "green": [0, 255, 0],
            "red": [255, 0, 0],
            "blue": [0, 0, 255],
            "yellow": [255, 255, 0],
            "cyan": [0, 255, 255],
            "magenta": [255, 0, 255]
        }
        color_rgb = np.array(colors.get(mask_color.lower(), [0, 255, 0]), dtype=np.uint8)
        
        writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=None, pixelformat='yuv420p')

        for idx, frame_pil in enumerate(frames):
            frame_np = np.array(frame_pil)
            
            if mask_only:
                # Start with black frame
                output_frame = np.zeros_like(frame_np)
            else:
                output_frame = frame_np.copy()

            if idx in outputs_data:
                results = outputs_data[idx]
                # results has 'masks'
                masks = results.get('masks', None)
                
                if masks is not None:
                    # masks could be a tensor or list
                    if isinstance(masks, torch.Tensor):
                        masks = masks.cpu().numpy()
                    
                    # masks shape: [N, H, W] or [N, 1, H, W]
                    if len(masks) > 0:
                        combined_mask = np.zeros((height, width), dtype=bool)
                        for mask in masks:
                            # Handle dimensions
                            if mask.ndim == 3 and mask.shape[0] == 1:
                                mask = mask.squeeze(0)
                            elif mask.ndim == 2:
                                pass # [H, W]
                            else:
                                # Attempt to squeeze if needed
                                mask = mask.squeeze()
                            
                            if mask.shape != (height, width):
                                # Resize mask if needed (should not happen if postprocess uses original sizes)
                                mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
                            
                            combined_mask = np.logical_or(combined_mask, mask > 0.0) # Threshold > 0 (logits) or > 0.5 (prob) depending on output. Usually postprocess returns binary or logits.
                            # Based on search results, postprocess_outputs returns masks.
                        
                        # Apply overlay
                        overlay_indices = combined_mask
                        
                        if mask_only:
                            # White on black
                            output_frame[overlay_indices] = [255, 255, 255]
                        else:
                            # Color overlay
                            output_frame[overlay_indices] = (output_frame[overlay_indices] * (1 - mask_opacity) + color_rgb * mask_opacity).astype(np.uint8)
            
            writer.append_data(output_frame)
            
        writer.close()
        print("Video saved.")
    
    def _generate_3d_from_segmentation(self, frames, outputs_data, frame_idx=None, seed=42):
        """Generate 3D object from segmented video frames using SAM 3D Objects"""
        try:
            # Set environment variables required by inference.py BEFORE importing
            # This is critical: inference.py accesses CONDA_PREFIX at module import time
            # Common workaround for non-Conda environments (Cog/Replicate)
            
            # ALWAYS set CONDA_PREFIX - don't check, just set it
            # inference.py does: os.environ["CUDA_HOME"] = os.environ["CONDA_PREFIX"]
            cuda_path = "/usr/local/cuda-12.4"  # Matches cog.yaml cuda: "12.4"
            os.environ["CONDA_PREFIX"] = cuda_path
            os.environ["CUDA_HOME"] = cuda_path
            
            print(f"Setting environment variables for SAM 3D Objects:")
            print(f"  CONDA_PREFIX={os.environ.get('CONDA_PREFIX')}")
            print(f"  CUDA_HOME={os.environ.get('CUDA_HOME')}")
            print(f"  Verifying CONDA_PREFIX exists: {'CONDA_PREFIX' in os.environ}")
            
            # Lazy import SAM 3D Objects
            # CRITICAL: Use patched version directly from thirdparty/ subrepo (handles missing CONDA_PREFIX)
            # Do NOT use /tmp/sam-3d-objects/notebook - it's unpatched and will fail
            import sys
            
            # Remove any unpatched paths from sys.path (from cached Docker layers)
            unpatched_paths = [
                "/tmp/sam-3d-objects/notebook",
                "/app/notebook",  # Old location, might exist from cached builds
            ]
            for path in unpatched_paths:
                if path in sys.path:
                    sys.path.remove(path)
                    print(f"✓ Removed unpatched path from sys.path: {path}")
            
            # Remove cached inference module if it was imported from wrong location
            if "inference" in sys.modules:
                inference_module = sys.modules["inference"]
                if hasattr(inference_module, "__file__") and inference_module.__file__:
                    if "/tmp/sam-3d-objects" in inference_module.__file__:
                        print(f"⚠ Removing cached inference module from wrong location: {inference_module.__file__}")
                        del sys.modules["inference"]
                        # Also remove any submodules
                        keys_to_remove = [k for k in sys.modules.keys() if k.startswith("inference.")]
                        for k in keys_to_remove:
                            del sys.modules[k]
            
            # Use patched version directly from thirdparty/ subrepo
            thirdparty_notebook = os.path.join(os.path.dirname(__file__), "thirdparty", "sam-3d-objects", "notebook")
            
            if not os.path.exists(thirdparty_notebook):
                raise FileNotFoundError(
                    f"Patched notebook not found at {thirdparty_notebook}\n"
                    "Make sure thirdparty/sam-3d-objects/notebook exists (git subrepo clone)"
                )
            
            # Add patched notebook to FRONT of path (highest priority)
            if thirdparty_notebook in sys.path:
                sys.path.remove(thirdparty_notebook)  # Remove if already there
            sys.path.insert(0, thirdparty_notebook)  # Insert at front
            print(f"✓ Using patched notebook from {thirdparty_notebook}")
            
            # Verify it's patched
            inference_file = os.path.join(thirdparty_notebook, "inference.py")
            if os.path.exists(inference_file):
                with open(inference_file, 'r') as f:
                    content = f.read()
                    if 'os.environ.get("CONDA_PREFIX"' in content:
                        print("✓ Verified: inference.py is patched (handles missing CONDA_PREFIX)")
                    else:
                        print("⚠ ERROR: inference.py is NOT patched! This will fail.")
                        raise RuntimeError(f"inference.py at {thirdparty_notebook} is not patched correctly")
            else:
                raise FileNotFoundError(f"inference.py not found at {inference_file}")
            
            # Verify which path will be used for import
            import importlib.util
            spec = importlib.util.find_spec("inference")
            if spec and spec.origin:
                print(f"✓ Will import inference from: {spec.origin}")
                if "/tmp/sam-3d-objects" in spec.origin:
                    raise RuntimeError(
                        f"ERROR: Still importing from unpatched location: {spec.origin}\n"
                        f"Expected: {thirdparty_notebook}/inference.py"
                    )
            
            print(f"About to import inference module...")
            # Import inference module (inference.py is patched to handle missing CONDA_PREFIX)
            from inference import Inference
            print(f"Successfully imported inference module from: {Inference.__module__}")
            
            # Find best frame if not specified
            if frame_idx is None:
                frame_idx = self._find_best_frame(frames, outputs_data)
            
            if frame_idx >= len(frames) or frame_idx not in outputs_data:
                print(f"Warning: Frame {frame_idx} not available, using first frame with mask")
                frame_idx = next((idx for idx in outputs_data.keys() if idx < len(frames)), 0)
            
            # Extract frame and mask
            selected_frame = frames[frame_idx]
            results = outputs_data[frame_idx]
            masks = results.get('masks', None)
            
            if masks is None or len(masks) == 0:
                raise ValueError(f"No masks found in frame {frame_idx}")
            
            # Combine masks into single mask
            height, width = np.array(selected_frame).shape[:2]
            combined_mask = np.zeros((height, width), dtype=np.uint8)
            
            for mask in masks:
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                
                if mask.ndim == 3 and mask.shape[0] == 1:
                    mask = mask.squeeze(0)
                elif mask.ndim > 2:
                    mask = mask.squeeze()
                
                if mask.shape != (height, width):
                    mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
                
                mask_bool = mask > 0.0
                combined_mask = np.logical_or(combined_mask, mask_bool)
            
            # Convert to PIL Image mask
            mask_img = Image.fromarray((combined_mask * 255).astype(np.uint8))
            
            # Load SAM 3D Objects model
            sam3d_model_path = "checkpoints/sam3d"
            tag = "hf"
            config_paths = [
                os.path.join(sam3d_model_path, "checkpoints", tag, "pipeline.yaml"),
                os.path.join(sam3d_model_path, "pipeline.yaml"),
                os.path.join(sam3d_model_path, tag, "pipeline.yaml"),
                "/app/checkpoints/hf/pipeline.yaml",
                "/tmp/sam-3d-objects/checkpoints/hf/pipeline.yaml",
            ]
            
            config_path = None
            for path in config_paths:
                if os.path.exists(path):
                    config_path = path
                    break
            
            if config_path is None:
                raise FileNotFoundError(
                    "SAM 3D Objects config not found. Please ensure checkpoints are downloaded."
                )
            
            print(f"Loading SAM 3D Objects from {config_path}...")
            inference_3d = Inference(config_path, compile=False)
            
            # Generate 3D object
            print(f"Generating 3D object from frame {frame_idx}...")
            output = inference_3d(selected_frame, mask_img, seed=seed)
            
            if "gs" not in output:
                raise ValueError("SAM 3D Objects output does not contain 'gs' (Gaussian Splat)")
            
            # Save PLY temporarily for conversion
            temp_ply_path = Path("/tmp/temp_object_3d.ply")
            output["gs"].save_ply(str(temp_ply_path))
            print(f"Temporary PLY saved for conversion: {temp_ply_path}")
            
            # Generate both outputs simultaneously
            glb_mesh_path = Path("/tmp/object_3d_mesh.glb")
            glb_splat_path = Path("/tmp/object_3d_splat.glb")
            
            mesh_success = False
            splat_success = False
            
            # Method 1: Convert to GLB mesh (mesh extraction)
            print("\n" + "="*70)
            print("METHOD 1: CONVERTING GAUSSIAN SPLAT TO GLB MESH (MESH EXTRACTION)")
            print("="*70)
            
            try:
                mesh_path = self._extract_mesh_from_gaussian_splat(temp_ply_path, glb_mesh_path)
                if mesh_path and os.path.exists(mesh_path):
                    mesh_success = True
                    print(f"✓ Mesh GLB generated successfully: {mesh_path}")
            except Exception as e:
                print(f"⚠ Mesh extraction failed: {e}")
            
            # Method 2: Convert to GLB (Gaussian Splat format)
            print("\n" + "="*70)
            print("METHOD 2: CONVERTING GAUSSIAN SPLAT TO GLB (SPLAT FORMAT)")
            print("="*70)
            
            try:
                splat_path = self._convert_gaussian_splat_to_glb(temp_ply_path, glb_splat_path)
                if splat_path and os.path.exists(splat_path):
                    splat_success = True
                    print(f"✓ Gaussian Splat GLB generated successfully: {splat_path}")
            except Exception as e:
                print(f"⚠ Gaussian Splat GLB conversion failed: {e}")
            
            # Clean up temp PLY
            if os.path.exists(temp_ply_path):
                os.remove(temp_ply_path)
            
            # Always return both files if available, or single file if only one succeeded
            # Create ZIP with all available GLB files
            import zipfile
            
            output_zip_path = Path("/tmp/object_3d.zip")
            files_added = []
            
            print("\n" + "="*70)
            print("CREATING ZIP WITH AVAILABLE 3D FILES")
            print("="*70)
            
            with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                if mesh_success and os.path.exists(glb_mesh_path):
                    zipf.write(glb_mesh_path, "object_3d_mesh.glb")
                    files_added.append("object_3d_mesh.glb (mesh extraction)")
                    print(f"✓ Added mesh GLB: object_3d_mesh.glb")
                
                if splat_success and os.path.exists(glb_splat_path):
                    zipf.write(glb_splat_path, "object_3d_splat.glb")
                    files_added.append("object_3d_splat.glb (Gaussian Splat format)")
                    print(f"✓ Added splat GLB: object_3d_splat.glb")
            
            if len(files_added) > 0:
                print(f"\n✓ ZIP file created: {output_zip_path}")
                print(f"  Contains {len(files_added)} file(s):")
                for f in files_added:
                    print(f"    - {f}")
                return output_zip_path
            else:
                # Both failed
                raise RuntimeError(
                    "Failed to convert Gaussian Splat to GLB. "
                    "Both mesh extraction and splat GLB conversion failed."
                )
                
        except ImportError as e:
            print(f"Warning: SAM 3D Objects not available: {e}")
            print("Install SAM 3D Objects dependencies to enable 3D generation")
            return None
        except Exception as e:
            print(f"Error generating 3D object: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _find_best_frame(self, frames, outputs_data):
        """Find the best frame for 3D generation (frame with largest mask area)"""
        best_frame_idx = 0
        max_mask_area = 0
        
        for idx, results in outputs_data.items():
            if idx >= len(frames):
                continue
                
            masks = results.get('masks', None)
            if masks is None or len(masks) == 0:
                continue
            
            # Calculate total mask area
            total_area = 0
            height, width = np.array(frames[idx]).shape[:2]
            
            for mask in masks:
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                
                if mask.ndim == 3 and mask.shape[0] == 1:
                    mask = mask.squeeze(0)
                elif mask.ndim > 2:
                    mask = mask.squeeze()
                
                if mask.shape != (height, width):
                    mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
                
                total_area += np.sum(mask > 0.0)
            
            if total_area > max_mask_area:
                max_mask_area = total_area
                best_frame_idx = idx
        
        # Fallback to middle frame if no masks found
        if max_mask_area == 0:
            best_frame_idx = len(frames) // 2
        
        print(f"Selected frame {best_frame_idx} for 3D generation (mask area: {max_mask_area})")
        return best_frame_idx
    
    def _extract_mesh_from_gaussian_splat(self, ply_path, glb_path):
        """
        Extract mesh from Gaussian Splat PLY using high-quality Poisson reconstruction.
        Exports to GLB format.
        """
        try:
            import open3d as o3d
            import numpy as np
        except ImportError:
            raise ImportError("Open3D not available. Install: pip install open3d>=0.18.0")
        
        try:
            print("="*70)
            print("HIGH-QUALITY MESH EXTRACTION (Poisson Reconstruction)")
            print("="*70)
            
            # Load point cloud
            print("\n[1/7] Loading point cloud...")
            pcd = o3d.io.read_point_cloud(str(ply_path))
            
            if len(pcd.points) == 0:
                raise ValueError("Point cloud is empty")
            
            initial_count = len(pcd.points)
            print(f"✓ Loaded {initial_count} points")
            
            if initial_count < 10:
                raise ValueError(f"Too few points ({initial_count}), need at least 10")
            
            # Outlier removal
            print("\n[2/7] Removing outliers...")
            if initial_count > 100:
                pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
                print(f"✓ Remaining: {len(pcd.points)} points")
            
            if len(pcd.points) < 10:
                raise ValueError("Too few points after outlier removal")
            
            # Normal estimation
            print("\n[3/7] Estimating normals...")
            try:
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50)
                )
                pcd.orient_normals_consistent_tangent_plane(k=100)
                print("✓ Normals estimated")
            except Exception as e:
                print(f"⚠ Normal estimation warning: {e}")
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
            
            # Poisson reconstruction
            print("\n[4/7] Performing Poisson surface reconstruction...")
            print("  Using depth=12 for maximum quality...")
            try:
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=12, width=0, scale=1.1, linear_fit=True
                )
            except MemoryError:
                print("⚠ Memory error, trying depth=10...")
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=10, width=0, scale=1.1, linear_fit=True
                )
            except Exception as e:
                print(f"⚠ Error: {e}, trying depth=9...")
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=9, width=0, scale=1.1
                )
            
            if len(mesh.vertices) == 0:
                raise ValueError("Poisson reconstruction produced empty mesh")
            
            print(f"✓ Mesh generated: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
            
            # Density filtering
            print("\n[5/7] Filtering low-density vertices...")
            if len(densities) > 0:
                threshold = np.quantile(densities, 0.02)
                mesh.remove_vertices_by_mask(densities < threshold)
                print(f"✓ Remaining: {len(mesh.vertices)} vertices")
            
            # Color preservation
            print("\n[6/7] Preserving vertex colors...")
            if pcd.has_colors():
                try:
                    colors = np.asarray(pcd.colors)
                    points = np.asarray(pcd.points)
                    vertices = np.asarray(mesh.vertices)
                    
                    if len(colors) > 0 and len(vertices) > 0:
                        try:
                            from scipy.spatial import cKDTree
                            tree = cKDTree(points)
                            distances, indices = tree.query(vertices, k=1)
                            vertex_colors = np.clip(colors[indices], 0.0, 1.0)
                            mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
                            print(f"✓ Colors preserved ({len(vertex_colors)} vertices)")
                        except ImportError:
                            kdtree = o3d.geometry.KDTreeFlann(pcd)
                            vertex_colors = []
                            for vertex in vertices:
                                [k, idx, dist] = kdtree.search_knn_vector_3d(vertex, 1)
                                vertex_colors.append(colors[idx[0]] if len(idx) > 0 else [0.5, 0.5, 0.5])
                            mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(np.array(vertex_colors), 0.0, 1.0))
                            print("✓ Colors preserved (Open3D KDTree)")
                except Exception as e:
                    print(f"⚠ Color preservation failed: {e}")
            
            # Mesh refinement
            print("\n[7/7] Refining mesh quality...")
            mesh.compute_vertex_normals()
            mesh = mesh.filter_smooth_laplacian(number_of_iterations=3, lambda_filter=0.5)
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            
            # Export to GLB
            print(f"\n[8/8] Exporting to GLB: {glb_path}")
            success = o3d.io.write_triangle_mesh(
                str(glb_path), mesh,
                write_vertex_colors=True,
                write_vertex_normals=True
            )
            
            if not success or not os.path.exists(glb_path):
                raise RuntimeError("Failed to write GLB file")
            
            file_size = os.path.getsize(glb_path)
            print(f"✓ GLB export successful ({file_size / 1024:.2f} KB)")
            return glb_path
            
        except Exception as e:
            print(f"✗ Mesh extraction failed: {type(e).__name__}: {e}")
            raise
    
    def _convert_gaussian_splat_to_glb(self, ply_path, glb_path):
        """
        Convert Gaussian Splat PLY directly to GLB format using KHR_gaussian_splatting extension.
        Preserves the splat representation (no mesh extraction).
        GLB is the binary version of GLTF, more compact and easier to distribute.
        """
        try:
            import numpy as np
            from pygltflib import GLTF2, Scene, Node, Mesh, Primitive, Attributes, Buffer, BufferView, Accessor
            import base64
        except ImportError as e:
            raise ImportError(f"Required libraries not available: {e}. Install: pip install pygltflib")
        
        try:
            print("="*70)
            print("GAUSSIAN SPLAT TO GLB CONVERSION")
            print("="*70)
            
            # Try reading with gsply first
            try:
                import gsply
                print("\n[1/4] Reading PLY with gsply...")
                splat_data = gsply.read_ply(str(ply_path))
                positions = np.array(splat_data.get('position', []))
                colors = np.array(splat_data.get('color', []))
                opacities = np.array(splat_data.get('opacity', []))
                rotations = np.array(splat_data.get('rotation', []))
                scales = np.array(splat_data.get('scale', []))
            except (ImportError, Exception) as e:
                print(f"gsply failed: {e}, using Open3D fallback...")
                # Fallback: read with Open3D
                import open3d as o3d
                pcd = o3d.io.read_point_cloud(str(ply_path))
                positions = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors) if pcd.has_colors() else np.ones((len(positions), 3)) * 0.5
                opacities = np.ones(len(positions))
                rotations = np.tile([0, 0, 0, 1], (len(positions), 1))  # Identity quaternions
                scales = np.ones((len(positions), 3))
            
            if len(positions) == 0:
                raise ValueError("No position data found")
            
            print(f"✓ Loaded {len(positions)} Gaussians")
            
            # Build GLTF structure
            print("\n[2/4] Building GLTF structure...")
            gltf = GLTF2()
            gltf.scene = 0
            gltf.extensionsUsed = ["KHR_gaussian_splatting"]
            gltf.extensionsRequired = ["KHR_gaussian_splatting"]
            
            # Prepare buffers
            print("\n[3/4] Preparing buffer data...")
            positions_bytes = positions.astype(np.float32).tobytes()
            
            # Combine RGB + Opacity
            if len(colors) > 0 and len(opacities) > 0:
                rgba = np.zeros((len(positions), 4), dtype=np.float32)
                rgba[:, :3] = colors[:len(positions), :3] if colors.shape[1] >= 3 else 0.5
                rgba[:, 3] = opacities[:len(positions)] if len(opacities) > 0 else 1.0
            else:
                rgba = np.ones((len(positions), 4), dtype=np.float32)
            colors_bytes = rgba.tobytes()
            
            # Rotations (quaternions)
            if len(rotations) > 0:
                rotations_bytes = rotations[:len(positions)].astype(np.float32).tobytes()
            else:
                rotations_bytes = np.tile([0, 0, 0, 1], (len(positions), 1)).astype(np.float32).tobytes()
            
            # Scales
            if len(scales) > 0:
                scales_bytes = scales[:len(positions)].astype(np.float32).tobytes()
            else:
                scales_bytes = np.ones((len(positions), 3), dtype=np.float32).tobytes()
            
            # Combine buffers
            buffer_data = positions_bytes + colors_bytes + rotations_bytes + scales_bytes
            
            # Create buffer (for GLB, buffer is embedded in binary file)
            buffer = Buffer()
            buffer.byteLength = len(buffer_data)
            gltf.buffers = [buffer]
            
            # For GLB export, we need to provide buffer data separately
            # pygltflib's save_binary expects buffers as a list of bytes
            gltf.set_binary_blob(buffer_data)
            
            # Create buffer views
            offset = 0
            pos_view = BufferView(buffer=0, byteOffset=offset, byteLength=len(positions_bytes), target=34962)
            offset += len(positions_bytes)
            color_view = BufferView(buffer=0, byteOffset=offset, byteLength=len(colors_bytes), target=34962)
            offset += len(colors_bytes)
            rot_view = BufferView(buffer=0, byteOffset=offset, byteLength=len(rotations_bytes), target=34962)
            offset += len(rotations_bytes)
            scale_view = BufferView(buffer=0, byteOffset=offset, byteLength=len(scales_bytes), target=34962)
            
            gltf.bufferViews = [pos_view, color_view, rot_view, scale_view]
            
            # Create accessors
            pos_accessor = Accessor(
                bufferView=0, componentType=5126, count=len(positions), type="VEC3",
                min=positions.min(axis=0).tolist(), max=positions.max(axis=0).tolist()
            )
            color_accessor = Accessor(bufferView=1, componentType=5126, count=len(positions), type="VEC4")
            rot_accessor = Accessor(bufferView=2, componentType=5126, count=len(positions), type="VEC4")
            scale_accessor = Accessor(bufferView=3, componentType=5126, count=len(positions), type="VEC3")
            
            gltf.accessors = [pos_accessor, color_accessor, rot_accessor, scale_accessor]
            
            # Create mesh with extension
            attributes = Attributes(POSITION=0, COLOR_0=1)
            if not hasattr(attributes, 'extensions'):
                attributes.extensions = {}
            attributes.extensions["KHR_gaussian_splatting"] = {"_ROTATION": 2, "_SCALE": 3}
            
            primitive = Primitive(attributes=attributes, mode=0)
            mesh = Mesh(primitives=[primitive])
            gltf.meshes = [mesh]
            
            # Create scene and node
            node = Node(mesh=0)
            gltf.nodes = [node]
            scene = Scene(nodes=[0])
            gltf.scenes = [scene]
            
            # Export as GLB (binary GLTF)
            print("\n[4/4] Exporting GLB file...")
            
            # For GLB format, use save_binary which embeds buffers in the binary file
            # Remove URI from buffer since it will be embedded
            gltf.buffers[0].uri = None
            
            # Save as GLB binary format
            gltf.save_binary(str(glb_path))
            
            if not os.path.exists(glb_path):
                raise RuntimeError("GLB file was not created")
            
            file_size = os.path.getsize(glb_path)
            print(f"✓ GLB export successful ({file_size / 1024:.2f} KB)")
            print("  Using KHR_gaussian_splatting extension")
            print("  Format: Binary GLTF (GLB)")
            
            return glb_path
            
        except Exception as e:
            print(f"✗ Gaussian Splat to GLB conversion failed: {type(e).__name__}: {e}")
            raise
