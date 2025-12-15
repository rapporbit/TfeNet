"""
Airway Segmentation Inference Script for Custom Datasets

This script performs airway segmentation on your own CT scans using the trained TfeNet model.

Usage:
    python inference.py --input /path/to/nifti/files --output /path/to/output --checkpoint /path/to/checkpoint

Arguments:
    --input: Path to folder containing NIfTI (.nii.gz or .nii) CT scan files
    --output: Path to save the segmentation results
    --checkpoint: Path to the checkpoint folder (default: ./checkpoint/ATM22)
    --dataset: Dataset checkpoint to use (choices: ATM22, BAS, AIIB23, default: ATM22)
    --use-small: Also use small airway model for better fine structure segmentation
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import SimpleITK as sitk
from importlib import import_module
from tqdm import tqdm
import warnings
from skimage import measure
from scipy import ndimage

warnings.filterwarnings("ignore")


def load_itk_image(filename):
    """Load ITK image and return numpy array with origin and spacing"""
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
    return numpyImage, numpyOrigin, numpySpacing


def save_itk(image, origin, spacing, filename):
    """Save numpy array as ITK image with origin and spacing"""
    if type(origin) != tuple:
        if type(origin) == list:
            origin = tuple(reversed(origin))
        else:
            origin = tuple(reversed(origin.tolist()))
    if type(spacing) != tuple:
        if type(spacing) == list:
            spacing = tuple(reversed(spacing))
        else:
            spacing = tuple(reversed(spacing.tolist()))
    itkimage = sitk.GetImageFromArray(image, isVector=False)
    itkimage.SetSpacing(spacing)
    itkimage.SetOrigin(origin)
    sitk.WriteImage(itkimage, filename, True)


def lumTrans_hu(img):
    """Apply Hounsfield Unit window clipping and normalization"""
    img[np.isnan(img)] = -2000
    lungwin = np.array([-1000., 600.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    newimg = (newimg * 255).astype('uint8')
    return newimg


def predict_single_case(image_path, model, cube_size=128, step=64, device='cuda'):
    """
    Predict airway segmentation for a single CT scan
    
    Args:
        image_path: Path to the NIfTI CT scan file
        model: Trained neural network model
        cube_size: Size of the cube for sliding window
        step: Step size for sliding window
        device: Device to run inference on ('cuda' or 'cpu')
        
    Returns:
        pred: Predicted segmentation mask
        origin: Image origin
        spacing: Voxel spacing
        case_name: Name of the case
    """
    # Load image
    case_name = os.path.basename(image_path).split('.nii')[0]
    print(f"Processing: {case_name}")
    
    imgs, origin, spacing = load_itk_image(image_path)
    
    # Apply HU windowing and normalization
    imgs = lumTrans_hu(imgs)
    imgs = (imgs.astype(np.float32)) / 255.0
    
    # Add batch and channel dimensions
    x = torch.from_numpy(imgs[np.newaxis, np.newaxis, ...]).float().to(device)
    
    # Initialize prediction arrays
    pred = np.zeros(x.shape)
    pred_num = np.zeros(x.shape)
    
    # Sliding window prediction
    xnum = (x.shape[2] - cube_size) // step + 1 if (x.shape[2] - cube_size) % step == 0 else \
        (x.shape[2] - cube_size) // step + 2
    ynum = (x.shape[3] - cube_size) // step + 1 if (x.shape[3] - cube_size) % step == 0 else \
        (x.shape[3] - cube_size) // step + 2
    znum = (x.shape[4] - cube_size) // step + 1 if (x.shape[4] - cube_size) % step == 0 else \
        (x.shape[4] - cube_size) // step + 2
    
    total_patches = xnum * ynum * znum
    print(f"  Image shape: {imgs.shape}")
    print(f"  Number of patches: {total_patches} ({xnum}x{ynum}x{znum})")
    
    with torch.no_grad():
        patch_idx = 0
        for xx in range(xnum):
            xl = step * xx
            xr = step * xx + cube_size
            if xr > x.shape[2]:
                xr = x.shape[2]
                xl = x.shape[2] - cube_size
            
            for yy in range(ynum):
                yl = step * yy
                yr = step * yy + cube_size
                if yr > x.shape[3]:
                    yr = x.shape[3]
                    yl = x.shape[3] - cube_size
                
                for zz in range(znum):
                    zl = step * zz
                    zr = step * zz + cube_size
                    if zr > x.shape[4]:
                        zr = x.shape[4]
                        zl = x.shape[4] - cube_size
                    
                    x_input = x[:, :, xl:xr, yl:yr, zl:zr]
                    p = model(x_input.contiguous())
                    p = p.cpu().detach().numpy()
                    pred[:, :, xl:xr, yl:yr, zl:zr] += p
                    pred_num[:, :, xl:xr, yl:yr, zl:zr] += 1
                    
                    patch_idx += 1
                    if patch_idx % max(1, total_patches // 10) == 0:
                        print(f"  Progress: {patch_idx}/{total_patches} patches ({100*patch_idx//total_patches}%)")
    
    # Average predictions and threshold
    pred = pred / pred_num
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    pred = np.squeeze(pred)
    
    return pred, origin, spacing, case_name


def postprocess_largest_component(pred):
    """Extract the largest connected component from prediction"""
    label, num = measure.label(pred, return_num=True, connectivity=1)
    if num == 0:
        return pred
    
    volume = np.zeros([num])
    for k in range(num):
        volume[k] = ((label == (k + 1)).astype(np.uint8)).sum()
    volume_sort = np.argsort(volume)
    
    large_cd = (label == (volume_sort[-1] + 1)).astype(np.uint8)
    large_cd = ndimage.binary_fill_holes(large_cd)
    
    return large_cd.astype('float')


def main():
    parser = argparse.ArgumentParser(
        description='Perform airway segmentation on custom CT scans using TfeNet'
    )
    parser.add_argument('--input', '-i', required=True,
                        help='Path to folder containing NIfTI CT scan files')
    parser.add_argument('--output', '-o', required=True,
                        help='Path to save segmentation results')
    parser.add_argument('--checkpoint', '-c', default='./checkpoint',
                        help='Path to checkpoint folder (default: ./checkpoint)')
    parser.add_argument('--dataset', '-d', default='ATM22',
                        choices=['ATM22', 'BAS', 'AIIB23'],
                        help='Dataset checkpoint to use (default: ATM22)')
    parser.add_argument('--use-small', action='store_true',
                        help='Also use small airway model and combine results')
    parser.add_argument('--no-postprocess', action='store_true',
                        help='Skip postprocessing (largest component extraction)')
    parser.add_argument('--device', default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to run inference on (default: cuda)')
    parser.add_argument('--cube-size', type=int, default=128,
                        help='Cube size for sliding window (default: 128)')
    parser.add_argument('--step', type=int, default=64,
                        help='Step size for sliding window (default: 64)')
    
    args = parser.parse_args()
    
    # Check if input path exists
    if not os.path.exists(args.input):
        raise ValueError(f"Input path does not exist: {args.input}")
    
    # Create output directories
    os.makedirs(args.output, exist_ok=True)
    temp_pred_path = os.path.join(args.output, 'temp_pred')
    os.makedirs(temp_pred_path, exist_ok=True)
    
    if args.use_small:
        temp_small_path = os.path.join(args.output, 'temp_small')
        os.makedirs(temp_small_path, exist_ok=True)
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        args.device = 'cpu'
    
    print(f"\n{'='*60}")
    print(f"TfeNet Airway Segmentation Inference")
    print(f"{'='*60}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print(f"Use small airway model: {args.use_small}")
    print(f"{'='*60}\n")
    
    # Load model
    print("Loading model...")
    casemodel = import_module('TfeNet')
    _, case_net = casemodel.get_model()
    
    # Load checkpoint for normal model
    checkpoint_path = os.path.join(args.checkpoint, args.dataset, 'TfeNet_checkpoint.ckpt')
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint not found: {checkpoint_path}\n"
                        f"Please download the checkpoint from the link in README.md")
    
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    case_net.load_state_dict(checkpoint['state_dict'])
    case_net = case_net.to(args.device)
    case_net.eval()
    print(f"Loaded model checkpoint: {checkpoint_path}")
    
    # Load small airway model if requested
    if args.use_small:
        small_checkpoint_path = os.path.join(args.checkpoint, args.dataset, 'TfeNetSmall_checkpoint.ckpt')
        if not os.path.exists(small_checkpoint_path):
            print(f"Warning: Small airway checkpoint not found: {small_checkpoint_path}")
            print("Proceeding without small airway model")
            args.use_small = False
        else:
            _, small_net = casemodel.get_model()
            small_checkpoint = torch.load(small_checkpoint_path, map_location=args.device)
            small_net.load_state_dict(small_checkpoint['state_dict'])
            small_net = small_net.to(args.device)
            small_net.eval()
            print(f"Loaded small airway model checkpoint: {small_checkpoint_path}")
    
    # Get list of NIfTI files
    file_list = [f for f in os.listdir(args.input) if f.endswith(('.nii', '.nii.gz'))]
    
    if len(file_list) == 0:
        raise ValueError(f"No NIfTI files found in {args.input}")
    
    print(f"\nFound {len(file_list)} CT scans to process\n")
    
    # Process each file
    for i, filename in enumerate(file_list):
        print(f"\n{'='*60}")
        print(f"Processing {i+1}/{len(file_list)}: {filename}")
        print(f"{'='*60}")
        
        image_path = os.path.join(args.input, filename)
        
        try:
            # Predict with normal model
            print("\nStep 1: Predicting with main model...")
            pred, origin, spacing, case_name = predict_single_case(
                image_path, case_net, cube_size=args.cube_size, 
                step=args.step, device=args.device
            )
            
            # Save temporary prediction
            temp_pred_file = os.path.join(temp_pred_path, f"{case_name}.nii.gz")
            save_itk(pred.astype('uint8'), origin, spacing, temp_pred_file)
            print(f"  Saved temporary prediction to {temp_pred_file}")
            
            # Predict with small airway model if requested
            if args.use_small:
                print("\nStep 2: Predicting with small airway model...")
                small_pred, _, _, _ = predict_single_case(
                    image_path, small_net, cube_size=args.cube_size,
                    step=args.step, device=args.device
                )
                
                # Save temporary small prediction
                temp_small_file = os.path.join(temp_small_path, f"{case_name}.nii.gz")
                save_itk(small_pred.astype('uint8'), origin, spacing, temp_small_file)
                print(f"  Saved temporary small prediction to {temp_small_file}")
                
                # Combine predictions
                print("\nStep 3: Combining predictions...")
                combined = pred + small_pred
                combined[combined > 0] = 1
                pred = combined.astype('uint8')
            
            # Postprocess if requested
            if not args.no_postprocess:
                step_num = 3 if args.use_small else 2
                print(f"\nStep {step_num}: Postprocessing (extracting largest component)...")
                pred = postprocess_largest_component(pred)
            
            # Save final result
            output_file = os.path.join(args.output, f"{case_name}.nii.gz")
            save_itk(pred.astype('uint8'), origin, spacing, output_file)
            print(f"\n✓ Saved final segmentation to: {output_file}")
            
        except Exception as e:
            print(f"\n✗ Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("Inference completed!")
    print(f"Results saved to: {args.output}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
