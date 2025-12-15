"""
DICOM to NIfTI Converter for Airway Segmentation

This script converts DICOM CT scans to NIfTI format (.nii.gz) which is required
for the TfeNet airway segmentation model.

Usage:
    python dicom_to_nifti.py --input /path/to/dicom/folder --output /path/to/output/folder

Arguments:
    --input: Path to the folder containing DICOM files (one folder per CT scan)
    --output: Path to save the converted NIfTI files
"""

import os
import argparse
import numpy as np
import SimpleITK as sitk
import pydicom as dicom
from glob import glob


def load_dicom_series(dicom_folder):
    """
    Load a DICOM series from a folder
    
    Args:
        dicom_folder: Path to folder containing DICOM files
        
    Returns:
        image: SimpleITK image object
        array: numpy array of the CT image
        origin: Image origin
        spacing: Voxel spacing
    """
    print(f"Loading DICOM series from: {dicom_folder}")
    
    # Read DICOM series
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder)
    
    if len(dicom_names) == 0:
        raise ValueError(f"No DICOM files found in {dicom_folder}")
    
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    
    # Get image properties
    array = sitk.GetArrayFromImage(image)
    origin = image.GetOrigin()
    spacing = image.GetSpacing()
    
    print(f"  Image shape: {array.shape}")
    print(f"  Spacing: {spacing}")
    print(f"  Origin: {origin}")
    
    return image, array, origin, spacing


def load_dicom_manual(dicom_folder):
    """
    Manually load DICOM files (alternative method)
    
    Args:
        dicom_folder: Path to folder containing DICOM files
        
    Returns:
        array: numpy array of the CT image
        origin: Image origin
        spacing: Voxel spacing
    """
    print(f"Loading DICOM files manually from: {dicom_folder}")
    
    slices = []
    for s in os.listdir(dicom_folder):
        if s.endswith('.dcm'):
            slices.append(dicom.read_file(os.path.join(dicom_folder, s), force=True))
    
    if len(slices) == 0:
        raise ValueError(f"No DICOM files found in {dicom_folder}")
    
    # Sort slices by ImagePositionPatient
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    
    # Handle duplicate slice positions
    if len(slices) > 1 and slices[0].ImagePositionPatient[2] == slices[1].ImagePositionPatient[2]:
        sec_num = 2
        while sec_num < len(slices) and slices[0].ImagePositionPatient[2] == slices[sec_num].ImagePositionPatient[2]:
            sec_num = sec_num + 1
        slice_num = int(len(slices) / sec_num)
        slices.sort(key=lambda x: float(x.InstanceNumber))
        slices = slices[0:slice_num]
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    
    # Calculate slice thickness
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except (AttributeError, IndexError):
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    
    for s in slices:
        s.SliceThickness = slice_thickness
    
    # Get pixel array
    image_array = np.stack([s.pixel_array for s in slices])
    image_array = image_array.astype(np.int16)
    
    # Convert to Hounsfield units
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image_array[slice_number] = slope * image_array[slice_number].astype(np.float64)
            image_array[slice_number] = image_array[slice_number].astype(np.int16)
        
        image_array[slice_number] += np.int16(intercept)
    
    # Get spacing and origin
    pixel_spacing = slices[0].PixelSpacing
    spacing = [slice_thickness, float(pixel_spacing[0]), float(pixel_spacing[1])]
    origin = slices[0].ImagePositionPatient
    
    print(f"  Image shape: {image_array.shape}")
    print(f"  Spacing: {spacing}")
    print(f"  Origin: {origin}")
    
    return image_array, origin, spacing


def save_as_nifti(array, origin, spacing, output_path):
    """
    Save numpy array as NIfTI file
    
    Args:
        array: numpy array of the CT image
        origin: Image origin
        spacing: Voxel spacing
        output_path: Path to save the NIfTI file
    """
    # Create SimpleITK image
    image = sitk.GetImageFromArray(array)
    
    # Set spacing and origin (reverse order for SimpleITK)
    if isinstance(spacing, (list, tuple)) and len(spacing) == 3:
        image.SetSpacing([spacing[2], spacing[1], spacing[0]])
    else:
        image.SetSpacing(spacing)
    
    if isinstance(origin, (list, tuple)) and len(origin) == 3:
        image.SetOrigin([origin[2], origin[1], origin[0]])
    else:
        image.SetOrigin(origin)
    
    # Save as NIfTI
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sitk.WriteImage(image, output_path, True)
    print(f"Saved NIfTI file to: {output_path}")


def convert_dicom_to_nifti(input_path, output_path, use_manual=False):
    """
    Convert DICOM CT scan to NIfTI format
    
    Args:
        input_path: Path to folder containing DICOM files
        output_path: Path to save the NIfTI file
        use_manual: Use manual DICOM loading method (default: False)
    """
    try:
        if use_manual:
            # Use manual loading method
            array, origin, spacing = load_dicom_manual(input_path)
            save_as_nifti(array, origin, spacing, output_path)
        else:
            # Use SimpleITK's GDCM reader
            try:
                image, array, origin, spacing = load_dicom_series(input_path)
                sitk.WriteImage(image, output_path, True)
                print(f"Saved NIfTI file to: {output_path}")
            except Exception as e:
                print(f"SimpleITK method failed: {e}")
                print("Trying manual loading method...")
                array, origin, spacing = load_dicom_manual(input_path)
                save_as_nifti(array, origin, spacing, output_path)
    except Exception as e:
        print(f"Error converting {input_path}: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Convert DICOM CT scans to NIfTI format for TfeNet airway segmentation'
    )
    parser.add_argument('--input', '-i', required=True,
                        help='Path to folder containing DICOM files or parent folder with multiple scans')
    parser.add_argument('--output', '-o', required=True,
                        help='Path to save converted NIfTI files')
    parser.add_argument('--manual', action='store_true',
                        help='Use manual DICOM loading method')
    parser.add_argument('--recursive', '-r', action='store_true',
                        help='Process all subfolders recursively')
    
    args = parser.parse_args()
    
    input_path = args.input
    output_path = args.output
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Check if input is a single DICOM series or parent folder with multiple series
    if args.recursive:
        # Process all subfolders
        print("Processing all subfolders recursively...")
        for root, dirs, files in os.walk(input_path):
            # Check if current folder contains DICOM files
            dicom_files = [f for f in files if f.endswith('.dcm')]
            if len(dicom_files) > 0:
                # Get relative path and create output filename
                rel_path = os.path.relpath(root, input_path)
                case_name = rel_path.replace(os.sep, '_')
                output_file = os.path.join(output_path, f"{case_name}.nii.gz")
                
                print(f"\n{'='*60}")
                print(f"Processing: {rel_path}")
                print(f"{'='*60}")
                
                try:
                    convert_dicom_to_nifti(root, output_file, use_manual=args.manual)
                except Exception as e:
                    print(f"Failed to convert {rel_path}: {e}")
                    continue
    else:
        # Check if input folder directly contains DICOM files
        dicom_files = [f for f in os.listdir(input_path) if f.endswith('.dcm')]
        
        if len(dicom_files) > 0:
            # Single DICOM series
            case_name = os.path.basename(input_path.rstrip('/\\'))
            output_file = os.path.join(output_path, f"{case_name}.nii.gz")
            
            print(f"\n{'='*60}")
            print(f"Processing single DICOM series: {case_name}")
            print(f"{'='*60}")
            
            convert_dicom_to_nifti(input_path, output_file, use_manual=args.manual)
        else:
            # Parent folder with multiple DICOM series
            print("Processing multiple DICOM series...")
            subfolders = [f for f in os.listdir(input_path) 
                         if os.path.isdir(os.path.join(input_path, f))]
            
            if len(subfolders) == 0:
                print("Error: No DICOM files or subfolders found in input path")
                return
            
            for subfolder in subfolders:
                subfolder_path = os.path.join(input_path, subfolder)
                output_file = os.path.join(output_path, f"{subfolder}.nii.gz")
                
                print(f"\n{'='*60}")
                print(f"Processing: {subfolder}")
                print(f"{'='*60}")
                
                try:
                    convert_dicom_to_nifti(subfolder_path, output_file, use_manual=args.manual)
                except Exception as e:
                    print(f"Failed to convert {subfolder}: {e}")
                    continue
    
    print(f"\n{'='*60}")
    print("Conversion completed!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
