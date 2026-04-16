# Simple OME-ZARR Plate Processor

Biaflows wrapper that applies Gaussian smoothing to OME-ZARR plate files and optionally performs a maximum intensity projection along Z.

All channels are preserved in the output. The output plate has the same structure and metadata as the input.

## Features

- Reads OME-ZARR plate format files (v0.4, Zarr v2) from the input directory
- Applies per-YX-plane Gaussian smoothing to every channel and Z-slice
- Optionally max-projects along the Z axis
- Writes output as new OME-ZARR plate files
- Multiple plates processed in parallel

## Parameters

- `--gaussian_sigma`: Sigma for Gaussian smoothing applied in YX to every plane. Set `0` to disable. (default: `2.0`)
- `--do_max_proj`: Perform max projection along Z axis — no-op when there is no Z axis (default: `true`)
- `--output_name`: Name prefix for output plates (default: `"processed"`)
- `--max_workers`: Number of parallel worker threads (default: `4`)

## Usage

### Local testing
```bash
python wrapper.py --infolder /path/to/input --outfolder /path/to/output --gaussian_sigma 2.0 --do_max_proj True
```

### Docker
```bash
docker build -t cellularimagingcf/simple-zarr-plate-processor .
docker run -v /input:/input -v /output:/output cellularimagingcf/simple-zarr-plate-processor \
  --infolder /input --outfolder /output --gaussian_sigma 2.0
```

## Input/Output

- **Input**: Directory containing `.zarr` files in OME-ZARR v0.4 format (Zarr v2, `.zattrs`/`.zarray` layout)
- **Output**: Processed `.zarr` files with naming pattern `{output_name}_{original_name}.zarr`

## Example

Smooth with sigma 3 and max-project Z:
```bash
python wrapper.py --infolder ./data --outfolder ./results --gaussian_sigma 3.0 --do_max_proj True --output_name "smoothed"
```

This will create files like `smoothed_plate001.zarr` from input `plate001.zarr`.

## Notes

- Only OME-ZARR v0.4 (Zarr v2) is supported. Files exported by `omero-cli-zarr 0.8+` default to v0.5/Zarr v3 — use `omero zarr --format 0.4 export` to produce compatible files.
- The output contains only the full-resolution level (level 0). Multiscale pyramids are not rebuilt.