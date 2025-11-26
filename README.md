# SAM 3 Video Segmentation

This is a [Replicate](https://replicate.com) implementation of **SAM 3**, a unified foundation model for promptable segmentation in images and videos from Meta Research.

This model allows you to segment objects in videos using text prompts, visual prompts (points/boxes), or simply track objects across frames.

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

## Citations

```bibtex
@article{sam3,
  title={SAM 3: Segment Anything Model 3},
  author={Meta Research},
  journal={arXiv preprint},
  year={2025}
}
```

## License

This model is built on top of the [SAM 3](https://github.com/facebookresearch/sam3) repository. Please refer to the original repository for license details regarding the model weights and code usage.

