# Minimal OME-ZARR plate processor
FROM python:3.11-slim

WORKDIR /app

# Install minimal dependencies for OME-ZARR processing
RUN pip install --no-cache-dir \
    zarr==2.18.2 \
    numpy==1.26.4 \
    ome-zarr==0.9.0 \
    dask[array]==2024.1.1 \
    numcodecs==0.12.1

# Copy application files
COPY . /app/

# Simple entrypoint
ENTRYPOINT ["python", "/app/wrapper.py"]
CMD []