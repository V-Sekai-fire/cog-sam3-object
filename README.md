# SAM 3 Video Segmentation

This is a [Replicate](https://replicate.com) implementation of **SAM 3**, a unified foundation model for promptable segmentation in images and videos from Meta Research.

This model allows you to segment objects in videos using text prompts, visual prompts (points/boxes), or simply track objects across frames. **You can also chain video segmentation into 3D object generation** - automatically create 3D models from segmented objects in your videos using SAM 3D Objects.

## Basic Usage

Run the model using `cog predict` on your local machine:

```bash
cog predict -i video=@foot.mp4 -i prompt="foot"
```

### Options

The model supports various output configurations and prompt types:

- **video**: Input video file path (required).
- **prompt**: Text description of the object to segment (e.g., "person", "car", "foot").
- **visual_prompt**: (Optional) JSON string for points/boxes (advanced usage).
- **mask_color**: Color of the segmentation overlay (default: "green"). Options: red, blue, yellow, cyan, magenta.
- **mask_opacity**: Opacity of the overlay (0.0 - 1.0).
- **mask_only**: If `True`, returns a high-contrast black & white mask video instead of an overlay.
- **return_zip**: If `True`, returns a ZIP file containing the overlay video AND individual PNG masks for every frame.
- **generate_3d**: If `True`, generates a 3D object (PLY file) from the segmented frames using SAM 3D Objects. The best frame with largest mask area is automatically selected.
- **frame_for_3d**: Optional frame index (0-based) to use for 3D generation. If not specified, automatically selects the best frame.
- **seed_3d**: Random seed for 3D object generation (default: 42).

## Examples

### 1. Basic Text Segmentation
Generate a green overlay for the "foot" in the video:

```bash
cog predict -i video=@foot.mp4 -i prompt="foot"
```

### 2. Custom Overlay Style
Generate a red overlay with 80% opacity:

```bash
cog predict -i video=@foot.mp4 \
    -i prompt="foot" \
    -i mask_color="red" \
    -i mask_opacity=0.8
```

### 3. Get Raw Masks for Editing
Get a ZIP file containing individual PNG masks for compositing in After Effects/Premiere:

```bash
cog predict -i video=@foot.mp4 \
    -i prompt="foot" \
    -i return_zip=True
```

### 4. Generate 3D Object from Video Segmentation
Chain video segmentation into 3D object generation. The model will automatically select the best frame and create a 3D model:

```bash
cog predict -i video=@foot.mp4 \
    -i prompt="foot" \
    -i generate_3d=True
```

### 5. Generate 3D Object from Specific Frame
Use a specific frame index for 3D generation:

```bash
cog predict -i video=@foot.mp4 \
    -i prompt="foot" \
    -i generate_3d=True \
    -i frame_for_3d=10
```

### 6. Complete Workflow: Video + Masks + 3D Model
Get everything in one ZIP file - segmented video, frame masks, and 3D model:

```bash
cog predict -i video=@foot.mp4 \
    -i prompt="foot" \
    -i return_zip=True \
    -i generate_3d=True
```

## Citations

```bibtex
@article{sam3,
  title={SAM 3: Segment Anything Model 3},
  author={Meta Research},
  journal={arXiv preprint},
  year={2025}
}
```

## 3D Object Generation

When `generate_3d=True`, the model:
1. Segments objects across all video frames (as usual)
2. Automatically selects the best frame (largest mask area) or uses the specified `frame_for_3d`
3. Extracts the combined mask from that frame
4. Uses [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) to generate a 3D Gaussian Splat (PLY file)
5. Returns the 3D model file (or includes it in the ZIP if `return_zip=True`)

**Requirements for 3D Generation:**
- SAM 3D Objects checkpoints must be available (downloaded automatically on first use)
- Additional dependencies may be required (see installation notes)

## License

This model is built on top of:
- [SAM 3](https://github.com/facebookresearch/sam3) for video segmentation
- [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) for 3D object generation

Please refer to the original repositories for license details regarding the model weights and code usage.

