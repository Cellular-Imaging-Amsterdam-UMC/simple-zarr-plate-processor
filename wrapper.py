#!/usr/bin/env python3
"""
Minimal OME-ZARR plate processor for biaflows

Simple processor that reads OME-ZARR plates, performs basic operations
(channel selection, max projection), and saves as OME-ZARR plates.
"""

import os
import sys
import logging
import argparse
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import zarr
import numpy as np
from ome_zarr import reader
from ome_zarr.writer import write_image
from ome_zarr.io import parse_url

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("zarr_plate_processor")


def find_zarr_plates(input_path):
    """Find all OME-ZARR plate files in the input directory"""
    plates = []
    input_path = Path(input_path)
    
    for item in input_path.iterdir():
        if item.is_dir() and item.name.endswith('.zarr'):
            # Check if it's a plate by looking for .zattrs with plate metadata
            zattrs_path = item / '.zattrs'
            if zattrs_path.exists():
                try:
                    store = zarr.DirectoryStore(str(item))
                    group = zarr.group(store=store)
                    if 'plate' in group.attrs:
                        plates.append(item)
                        logger.info(f"Found OME-ZARR plate: {item.name}")
                    else:
                        logger.info(f"Found OME-ZARR (not plate): {item.name}")
                        plates.append(item)  # Include anyway for processing
                except Exception as e:
                    logger.warning(f"Could not read {item.name}: {e}")
    
    logger.info(f"Found {len(plates)} ZARR files to process")
    return plates


def process_zarr_data(data_array, channel=-1, do_max_proj=True):
    """
    Process the image data array
    
    Args:
        data_array: Input image data (assumed to be in TCZYX or similar format)
        channel: Channel to extract (-1 for all channels)
        do_max_proj: Whether to do max projection along Z
    
    Returns:
        Processed array
    """
    logger.info(f"Input array shape: {data_array.shape}")
    
    # Assume data is in TCZYX format (time, channel, z, y, x)
    # or CZYX format (channel, z, y, x)
    
    # Handle different dimensionalities
    if len(data_array.shape) == 5:  # TCZYX
        t, c, z, y, x = data_array.shape
        logger.info(f"Data format: TCZYX ({t}, {c}, {z}, {y}, {x})")
        
        # Channel selection
        if channel >= 0 and channel < c:
            logger.info(f"Selecting channel {channel}")
            processed = data_array[:, channel:channel+1, :, :, :]  # Keep channel dim
        else:
            logger.info("Keeping all channels")
            processed = data_array
        
        # Max projection along Z if requested
        if do_max_proj and z > 1:
            logger.info("Performing max projection along Z axis")
            processed = np.max(processed, axis=2, keepdims=True)  # Keep Z dim with size 1
        
    elif len(data_array.shape) == 4:  # CZYX
        c, z, y, x = data_array.shape
        logger.info(f"Data format: CZYX ({c}, {z}, {y}, {x})")
        
        # Channel selection
        if channel >= 0 and channel < c:
            logger.info(f"Selecting channel {channel}")
            processed = data_array[channel:channel+1, :, :, :]  # Keep channel dim
        else:
            logger.info("Keeping all channels")
            processed = data_array
        
        # Max projection along Z if requested
        if do_max_proj and z > 1:
            logger.info("Performing max projection along Z axis")
            processed = np.max(processed, axis=1, keepdims=True)  # Keep Z dim with size 1
        
    elif len(data_array.shape) == 3:  # ZYX or CYX
        if do_max_proj:
            logger.info("Performing max projection along first axis")
            processed = np.max(data_array, axis=0, keepdims=True)
        else:
            processed = data_array
    else:
        logger.info("Data shape not recognized, returning as-is")
        processed = data_array
    
    logger.info(f"Output array shape: {processed.shape}")
    return processed


def process_single_zarr(zarr_path, output_dir, channel=-1, do_max_proj=True, output_name="processed"):
    """Process a single OME-ZARR file"""
    logger.info(f"[{zarr_path.name}] Starting processing...")
    
    # Set up output path
    output_zarr = output_dir / f"{output_name}_{zarr_path.name}"
    if output_zarr.exists():
        shutil.rmtree(output_zarr)
    
    try:
        # Create proper zarr store for input
        input_store = zarr.storage.FSStore(str(zarr_path))
        
        # Check if this is a plate format
        root_group = zarr.open_group(store=input_store, mode='r')
        if 'plate' in root_group.attrs:
            logger.info(f"[{zarr_path.name}] Processing OME-ZARR plate format")
            return process_plate_format(zarr_path, output_zarr, root_group, channel, do_max_proj)
        else:
            logger.info(f"[{zarr_path.name}] Processing single image format")
            # For non-plate format, fall back to ome-zarr reader
            store = parse_url(str(zarr_path), mode="r")
            reader_obj = reader.Reader(store)
            return process_single_image_format(zarr_path, output_zarr, reader_obj, channel, do_max_proj)
            
    except Exception as e:
        error_msg = f"[{zarr_path.name}] Failed to process: {str(e)}"
        logger.error(error_msg)
        return {"status": "error", "file": zarr_path.name, "error": str(e)}


def process_plate_format(zarr_path, output_zarr, root_group, channel, do_max_proj):
    """Process OME-ZARR plate format with wells and fields"""
    plate_info = root_group.attrs['plate']
    wells = plate_info.get('wells', [])
    
    logger.info(f"[{zarr_path.name}] Found {len(wells)} wells to process")
    
    # Create output store using proper FSStore
    output_store = zarr.storage.FSStore(str(output_zarr))
    output_root = zarr.group(store=output_store, overwrite=True)
    
    # Copy plate metadata
    output_root.attrs.update(root_group.attrs)
    
    # Process each well
    for well_info in wells:
        well_path = well_info['path']
        logger.info(f"[{zarr_path.name}] Processing well: {well_path}")
        
        try:
            # Read well group
            well_group = root_group[well_path]
            well_attrs = well_group.attrs
            
            # Create output well group
            output_well = output_root.create_group(well_path)
            output_well.attrs.update(well_attrs)
            
            # Process each field/image in the well
            well_data = well_attrs.get('well', {})
            images = well_data.get('images', [])
            
            for img_info in images:
                field_path = img_info['path']
                logger.info(f"[{zarr_path.name}] Processing field: {well_path}/{field_path}")
                
                # Read the field image data
                field_group = well_group[field_path]
                
                # Get the highest resolution (usually level 0)
                data_array = np.array(field_group['0'])  # Level 0 = highest resolution
                logger.info(f"[{zarr_path.name}] Field {well_path}/{field_path} shape: {data_array.shape}")
                
                # Process the field data
                processed_data = process_zarr_data(data_array, channel=channel, do_max_proj=do_max_proj)
                
                # Create output field group with proper single-scale metadata
                output_field = output_well.create_group(field_path)
                
                # Copy base attributes but update multiscales for single resolution
                field_attrs = dict(field_group.attrs)
                
                # Update multiscales metadata for single resolution level
                if 'multiscales' in field_attrs:
                    multiscales = field_attrs['multiscales'][0].copy()  # Take first multiscale
                    # Keep only the first dataset (level 0)
                    multiscales['datasets'] = [multiscales['datasets'][0]]
                    field_attrs['multiscales'] = [multiscales]
                
                output_field.attrs.update(field_attrs)
                
                # Write processed data with proper zarr array metadata
                # Get original array metadata from level 0
                original_array = field_group['0']
                original_meta = dict(original_array.attrs)
                
                # Create dataset with same chunking strategy and metadata as original
                output_data = output_field.create_dataset(
                    '0', 
                    data=processed_data, 
                    chunks=True,
                    dtype=processed_data.dtype,
                    compressor=original_array.compressor,
                    fill_value=original_array.fill_value,
                    order=original_array.order,
                    filters=original_array.filters
                )
                
                # Copy array-level attributes
                output_data.attrs.update(original_meta)
                
                # Validate that the zarr array was created properly
                try:
                    # Force zarr to write metadata by accessing the array info
                    logger.info(f"[{zarr_path.name}] Created zarr array - shape: {output_data.shape}, dtype: {output_data.dtype}, chunks: {output_data.chunks}")
                except Exception as e:
                    logger.warning(f"[{zarr_path.name}] Warning during zarr validation: {e}")
                
                logger.info(f"[{zarr_path.name}] ✓ Processed field {well_path}/{field_path}")
        
        except Exception as e:
            logger.warning(f"[{zarr_path.name}] Failed to process well {well_path}: {str(e)}")
            continue
    
    # Ensure all data is written to disk
    try:
        output_store.close()
        logger.info(f"[{zarr_path.name}] Synchronized zarr store to disk")
    except Exception as e:
        logger.warning(f"[{zarr_path.name}] Store sync warning: {e}")
    
    logger.info(f"[{zarr_path.name}] Successfully processed plate!")
    return {"status": "success", "file": zarr_path.name, "output": output_zarr}


def process_single_image_format(zarr_path, output_zarr, reader_obj, channel, do_max_proj):
    """Process single OME-ZARR image (non-plate format)"""
    # Get the first (and usually only) image
    nodes = list(reader_obj())
    if not nodes:
        raise ValueError("No image data found in ZARR file")
    
    image_node = nodes[0]
    image_data = image_node.data
    
    # Get image data as numpy array (load into memory for processing)
    logger.info(f"[{zarr_path.name}] Loading image data...")
    if hasattr(image_data, 'compute'):  # dask array
        data_array = image_data.compute()
    else:
        data_array = np.asarray(image_data)
    
    # Process the data
    logger.info(f"[{zarr_path.name}] Processing data...")
    processed_data = process_zarr_data(data_array, channel=channel, do_max_proj=do_max_proj)
    
    # Write processed data back as OME-ZARR
    logger.info(f"[{zarr_path.name}] Writing processed data to {output_zarr}")
    
    # Create output store
    output_store = parse_url(str(output_zarr), mode="w")
    
    # Write the image with basic metadata
    write_image(
        image=processed_data,
        group=output_store,
        axes="tczyx" if len(processed_data.shape) == 5 else "czyx" if len(processed_data.shape) == 4 else None,
        scaler=None,  # No multiscale for simplicity
    )
    
    logger.info(f"[{zarr_path.name}] Successfully processed!")
    return {"status": "success", "file": zarr_path.name, "output": output_zarr}


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Simple OME-ZARR plate processor")
    
    # Standard biaflows parameters
    parser.add_argument("--infolder", required=True, help="Input folder containing OME-ZARR plates")
    parser.add_argument("--outfolder", required=True, help="Output folder for processed plates")
    parser.add_argument("--gtfolder", help="Ground truth folder (unused)")
    parser.add_argument("--local", action="store_true", help="Local mode (compatibility)")
    parser.add_argument("-nmc", action="store_true", help="No model cache (compatibility)")
    
    # Processing parameters
    parser.add_argument("--channel", type=int, default=0, 
                       help="Channel to extract/process (0-based), -1 for all channels")
    parser.add_argument("--do_max_proj", action="store_true", default=True,
                       help="Perform maximum intensity projection along Z axis")
    parser.add_argument("--output_name", type=str, default="processed",
                       help="Name prefix for output plates")
    parser.add_argument("--max_workers", type=int, default=4,
                       help="Maximum number of parallel worker threads (default: 4)")
    
    return parser.parse_args()


def main():
    """Main processing function"""
    logger.info("Starting OME-ZARR plate processor")
    
    # Parse arguments
    args = parse_args()
    logger.info(f"Arguments: {args}")
    
    try:
        # Set up paths
        input_path = Path(args.infolder)
        output_path = Path(args.outfolder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Input path: {input_path}")
        logger.info(f"Output path: {output_path}")
        
        # Find OME-ZARR plates
        zarr_plates = find_zarr_plates(input_path)
        
        if not zarr_plates:
            logger.warning("No OME-ZARR plates found in input directory")
            return
        
        # Process plates in parallel
        logger.info(f"Processing {len(zarr_plates)} plates using {args.max_workers} parallel workers")
        
        results = []
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # Submit all tasks
            future_to_zarr = {
                executor.submit(
                    process_single_zarr,
                    zarr_file,
                    output_path,
                    channel=args.channel,
                    do_max_proj=args.do_max_proj,
                    output_name=args.output_name
                ): zarr_file for zarr_file in zarr_plates
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_zarr):
                zarr_file = future_to_zarr[future]
                try:
                    result = future.result()
                    results.append(result)
                    if result["status"] == "success":
                        logger.info(f"✓ Completed processing: {result['file']}")
                    else:
                        logger.error(f"✗ Failed processing: {result['file']} - {result['error']}")
                except Exception as e:
                    error_result = {"status": "error", "file": zarr_file.name, "error": str(e)}
                    results.append(error_result)
                    logger.error(f"✗ Exception processing {zarr_file.name}: {str(e)}")
        
        # Final status
        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] == "error"]
        
        logger.info(f"\n=== Processing Summary ===\n"
                   f"Total plates: {len(zarr_plates)}\n"
                   f"Successful: {len(successful)}\n"
                   f"Failed: {len(failed)}\n")
        
        if failed:
            logger.error("Failed plates:")
            for fail in failed:
                logger.error(f"  - {fail['file']}: {fail['error']}")
            logger.error("\nProcessing completed with errors!")
            sys.exit(1)
        else:
            logger.info("All plates processed successfully!")
            logger.info("Output files:")
            for success in successful:
                logger.info(f"  - {success['output']}")
    
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()