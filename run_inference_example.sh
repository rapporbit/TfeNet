#!/bin/bash
# Example script for running airway segmentation on custom DICOM data
# 
# Usage: bash run_inference_example.sh
#
# Before running this script:
# 1. Make sure you have installed all dependencies (see INFERENCE_GUIDE.md)
# 2. Download the pre-trained models to ./checkpoint directory
# 3. Update the DICOM_INPUT_PATH variable below to point to your DICOM data

set -e  # Exit on error

# Configuration
DICOM_INPUT_PATH="./example_dicom_data"  # Update this to your DICOM data path
NIFTI_OUTPUT_PATH="./converted_nifti"
SEGMENTATION_OUTPUT_PATH="./segmentation_results"
DATASET="ATM22"  # Options: ATM22, BAS, AIIB23
USE_SMALL_MODEL=true  # Set to false to skip small airway model

echo "================================================================"
echo "TfeNet Airway Segmentation - Complete Pipeline"
echo "================================================================"
echo ""

# Check if DICOM input path exists
if [ ! -d "$DICOM_INPUT_PATH" ]; then
    echo "Error: DICOM input path does not exist: $DICOM_INPUT_PATH"
    echo "Please update DICOM_INPUT_PATH in this script to point to your DICOM data"
    echo ""
    echo "Example DICOM data structure:"
    echo "  $DICOM_INPUT_PATH/"
    echo "  ├── patient001/"
    echo "  │   ├── IM0001.dcm"
    echo "  │   ├── IM0002.dcm"
    echo "  │   └── ..."
    echo "  ├── patient002/"
    echo "  │   └── ..."
    exit 1
fi

# Check if checkpoint exists
if [ ! -d "./checkpoint/$DATASET" ]; then
    echo "Error: Checkpoint directory not found: ./checkpoint/$DATASET"
    echo "Please download the pre-trained models from:"
    echo "https://drive.google.com/file/d/1DEKyAMhV90AL80qoy2QgDxmO0F1cXx1r/view?usp=drive_link"
    echo ""
    echo "Extract to ./checkpoint directory"
    exit 1
fi

echo "Step 1: Converting DICOM to NIfTI format"
echo "----------------------------------------------------------------"
python dicom_to_nifti.py \
    --input "$DICOM_INPUT_PATH" \
    --output "$NIFTI_OUTPUT_PATH"

echo ""
echo "Step 2: Running airway segmentation"
echo "----------------------------------------------------------------"

# Build inference command
INFERENCE_CMD="python inference.py \
    --input $NIFTI_OUTPUT_PATH \
    --output $SEGMENTATION_OUTPUT_PATH \
    --dataset $DATASET"

# Add small model flag if enabled
if [ "$USE_SMALL_MODEL" = true ]; then
    INFERENCE_CMD="$INFERENCE_CMD --use-small"
fi

# Run inference
$INFERENCE_CMD

echo ""
echo "================================================================"
echo "Airway Segmentation Completed Successfully!"
echo "================================================================"
echo ""
echo "Results saved to: $SEGMENTATION_OUTPUT_PATH"
echo ""
echo "You can view the results using:"
echo "  - ITK-SNAP: https://www.itksnap.org/"
echo "  - 3D Slicer: https://www.slicer.org/"
echo "  - MITK Workbench: https://www.mitk.org/"
echo ""
