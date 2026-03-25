# Simple OME-ZARR Plate Processor

Minimal biaflows wrapper for processing OME-ZARR plate format files.

## Features

- Reads OME-ZARR plate format files from input directory
- Performs simple operations:
  - Channel selection (extract specific channel or keep all)
  - Maximum intensity projection along Z axis
- Writes processed data as new OME-ZARR plate files
- Minimal dependencies (zarr, numpy, ome-zarr, dask)

## Parameters

- `--channel`: Channel index to process (0-based), -1 for all channels (default: 0)
- `--do_max_proj`: Perform max projection along Z axis (default: true)  
- `--output_name`: Name prefix for output plates (default: "processed")

## Usage

### Local testing
```bash
python wrapper.py --infolder /path/to/input --outfolder /path/to/output --channel 0 --do_max_proj
```

### Docker
```bash
docker build -t biaflows/simple-zarr-plate-processor .
docker run -v /input:/input -v /output:/output biaflows/simple-zarr-plate-processor \
  --infolder /input --outfolder /output --channel 0 --do_max_proj
```

## Input/Output

- **Input**: Directory containing `.zarr` files in OME-ZARR format
- **Output**: Processed `.zarr` files with naming pattern `{output_name}_{original_name}.zarr`

## Example

Process channel 1 with max projection:
```bash
python wrapper.py --infolder ./data --outfolder ./results --channel 1 --do_max_proj --output_name "maxproj"
```

This will create files like `maxproj_plate001.zarr` from input `plate001.zarr`.