# Simple OME-ZARR Plate Processor

Biaflows wrapper that applies **Sobel edge detection** to OME-ZARR plate files. Structures appear as bright outlines — the effect is unmistakably visible. Optionally pre-smooths with Gaussian to reduce noise, max-projects along Z, and normalizes per-channel contrast.

All channels are preserved in the output. The output plate has the same structure and metadata as the input.

## Features

- Reads OME-ZARR plate format files (v0.4, Zarr v2) from the input directory
- Applies Sobel edge detection to every YX plane across all channels and Z-slices
- Optional Gaussian pre-smoothing before edge detection (reduces noise artefacts in edges)
- Normalizes contrast per channel (2nd–98th percentile stretch to full dtype range)
- Optionally max-projects along the Z axis
- Writes output as new OME-ZARR plate files
- Multiple plates processed in parallel

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `--gaussian_sigma` | `1.0` | Gaussian pre-smoothing sigma applied before edge detection. Set `0` to skip. |
| `--do_max_proj` | `true` | Max-project along Z axis (no-op if no Z axis present). |
| `--normalize_contrast` | `true` | Per-channel 2nd–98th percentile stretch to full dtype range. |
| `--output_name` | `"processed"` | Name prefix for output plates. |
| `--max_workers` | `4` | Number of parallel worker threads. |

## Usage

### Local testing
```bash
python wrapper.py --infolder /path/to/input --outfolder /path/to/output
```

### Docker
```bash
docker build -t cellularimagingcf/simple-zarr-plate-processor .
docker run -v /input:/input -v /output:/output cellularimagingcf/simple-zarr-plate-processor \
  --infolder /input --outfolder /output
```

## Input/Output

- **Input**: Directory containing `.zarr` files in OME-ZARR v0.4 format (Zarr v2, `.zattrs`/`.zarray` layout)
- **Output**: Processed `.zarr` files with naming pattern `{output_name}_{original_name}.zarr`

## Notes

- Only OME-ZARR v0.4 (Zarr v2) is supported. Files exported by `omero-cli-zarr 0.8+` default to v0.5/Zarr v3 — use `omero zarr --format 0.4 export` to produce compatible files.
- The output contains only the full-resolution level (level 0). Multiscale pyramids are not rebuilt.
- Contrast normalization runs after edge detection and max-projection.