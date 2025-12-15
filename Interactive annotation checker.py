"""
Interactive Passport Annotation Checker - FILE OUTPUT VERSION
=======================================
Saves annotation visualizations to files instead of displaying them
Perfect for systems without display support (headless/remote)
"""

import json
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from datetime import datetime

def check_single_annotation(base_path, country="srb_passport", folder_idx=0, img_idx=0, output_dir=None):
    """
    Check a single annotation and save visualization to file
    
    Args:
        base_path: Root directory containing annotation and image folders
        country: Country folder name (e.g., "srb_passport")
        folder_idx: Index of the annotation folder to check
        img_idx: Index of the image within that folder
        output_dir: Directory to save output images (default: base_path/annotation_checks)
    
    Returns:
        Path to saved image if successful, None otherwise
    """
    base_path = Path(base_path)
    
    if output_dir is None:
        output_dir = base_path / "annotation_checks"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Get annotation file
    annotation_dir = base_path / "annotation" / country
    json_files = sorted(list(annotation_dir.glob("*.json")))
    
    if not json_files:
        print(f" No JSON files found in {annotation_dir}")
        return None
    
    if folder_idx >= len(json_files):
        print(f" Folder index {folder_idx} out of range. Max: {len(json_files)-1}")
        return None
    
    json_file = json_files[folder_idx]
    folder_name = json_file.stem
    
    print(f" Checking: {country}/{folder_name}")
    
    # Load annotation (VIA format)
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract VIA metadata
    annotations = data.get('_via_img_metadata', {})
    
    if not annotations:
        print(f"  No _via_img_metadata found in {json_file.name}")
        return None
    
    # Get image files
    img_folder = base_path / "images" / country / folder_name
    img_files = sorted(list(img_folder.glob("*.jpg")) + list(img_folder.glob("*.png")))
    
    if not img_files:
        print(f" No images found in {img_folder}")
        return None
    
    if img_idx >= len(img_files):
        print(f" Image index {img_idx} out of range. Max: {len(img_files)-1}")
        return None
    
    img_path = img_files[img_idx]
    
    # Find annotation for this image
    annotation_found = False
    points = []
    
    for key, ann_data in annotations.items():
        if isinstance(ann_data, dict) and ann_data.get('filename') == img_path.name:
            annotation_found = True
            
            # Extract polygon points
            for region in ann_data.get('regions', []):
                shape = region.get('shape_attributes', {})
                region_attrs = region.get('region_attributes', {})
                
                # Check for field_name (VIA format)
                if (shape.get('name') == 'polygon' and 
                    region_attrs.get('field_name') == 'doc_quad'):
                    
                    x_points = shape.get('all_points_x', [])
                    y_points = shape.get('all_points_y', [])
                    
                    if len(x_points) == len(y_points):
                        points = list(zip(x_points, y_points))
                        break
            
            break
    
    if not annotation_found:
        print(f" No annotation found for {img_path.name}")
        return None
    
    # Visualize
    img = cv2.imread(str(img_path))
    if img is None:
        print(f" Cannot read image: {img_path}")
        return None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.imshow(img_rgb)
    
    if points:
        # Draw polygon
        points_np = np.array(points + [points[0]], dtype=np.int32)
        ax.plot(points_np[:, 0], points_np[:, 1], 'g-', linewidth=3, label='Document boundary')
        
        # Draw corner points with labels
        for i, (x, y) in enumerate(points):
            ax.plot(x, y, 'ro', markersize=10)
            ax.text(x+10, y-10, f'P{i+1}', color='yellow', fontsize=12, 
                   fontweight='bold', 
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='red', alpha=0.8))
        
        # Calculate and show statistics
        area = cv2.contourArea(np.array(points, dtype=np.float32))
        img_area = img.shape[0] * img.shape[1]
        area_percentage = (area / img_area) * 100
        
        # Calculate aspect ratio
        rect = cv2.boundingRect(np.array(points, dtype=np.float32))
        _, _, rect_w, rect_h = rect
        aspect_ratio = rect_w / rect_h if rect_h > 0 else 0
        
        info_text = f"Image: {img_path.name}\n"
        info_text += f"Size: {img.shape[1]}×{img.shape[0]} pixels\n"
        info_text += f"Polygon vertices: {len(points)}\n"
        info_text += f"Document area: {area_percentage:.1f}% of image\n"
        info_text += f"Aspect ratio: {aspect_ratio:.2f}"
        
        ax.text(20, 50, info_text, 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=2),
               fontsize=11, color='black', family='monospace', verticalalignment='top')
        
        ax.legend(loc='upper right', fontsize=12)
        
        status_text = "ANNOTATION FOUND"
        status_color = 'green'
    else:
        ax.text(img.shape[1]//2, img.shape[0]//2, 
               'NO ANNOTATION FOUND', 
               ha='center', va='center', color='red', fontsize=24,
               fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9, edgecolor='red', linewidth=3))
        
        status_text = "NO ANNOTATION"
        status_color = 'red'
    
    ax.set_title(f"{country} / {folder_name} / {img_path.name}\n{status_text}", 
                fontsize=14, fontweight='bold', color=status_color, pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    
    # Save to file instead of showing
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{country}_{folder_name}_img{img_idx:03d}_{timestamp}.png"
    output_path = output_dir / output_filename
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved visualization: {output_path}")
    
    return output_path


def quick_check_serbian(base_path, num_folders=3, num_images=3, output_dir=None):
    """
    Quick check for Serbian passport annotations
    Saves visualizations for the first few images from the first few folders
    
    Args:
        base_path: Root directory
        num_folders: Number of folders to check (default: 3)
        num_images: Number of images per folder to check (default: 3)
        output_dir: Where to save output images
    """
    print("\n" + "="*70)
    print("QUICK CHECK - Serbian Passport Annotations")
    print("="*70)
    print(f"Checking first {num_images} images from first {num_folders} folders...")
    print(f"Visualizations will be saved to: {output_dir or Path(base_path) / 'annotation_checks'}\n")
    
    base_path = Path(base_path)
    
    if output_dir is None:
        output_dir = base_path / "annotation_checks"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Get available folders
    annotation_dir = base_path / "annotation" / "srb_passport"
    json_files = sorted(list(annotation_dir.glob("*.json")))
    
    if not json_files:
        print("No annotation files found for Serbian passports")
        return
    
    total_checked = 0
    saved_files = []
    total_folders = min(num_folders, len(json_files))
    
    # Check first N images from first M folders
    for folder_idx in range(total_folders):
        folder_name = json_files[folder_idx].stem
        img_folder = base_path / "images" / "srb_passport" / folder_name
        img_count = len(list(img_folder.glob("*.jpg")) + list(img_folder.glob("*.png")))
        total_images = min(num_images, img_count)
        
        print(f"\n Folder {folder_idx+1}/{total_folders}: {folder_name} ({img_count} images)")
        
        for img_idx in range(total_images):
            print(f"     Image {img_idx+1}/{total_images}", end=" ")
            result = check_single_annotation(base_path, "srb_passport", folder_idx, img_idx, output_dir)
            
            if result:
                saved_files.append(result)
                total_checked += 1
    
    print(f"\n" + "="*70)
    print(f" Quick check completed!")
    print(f" Total images checked: {total_checked}")
    print(f" Saved {len(saved_files)} visualizations to: {output_dir}")
    print("="*70)
    
    return saved_files


def check_random_samples(base_path, country="srb_passport", num_samples=5, output_dir=None):
    """
    Check random samples from the dataset and save visualizations
    
    Args:
        base_path: Root directory
        country: Country to check
        num_samples: Number of random images to check
        output_dir: Where to save output images
    """
    import random
    
    base_path = Path(base_path)
    
    if output_dir is None:
        output_dir = base_path / "annotation_checks"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    annotation_dir = base_path / "annotation" / country
    json_files = sorted(list(annotation_dir.glob("*.json")))
    
    if not json_files:
        print(f" No annotation files found for {country}")
        return
    
    print("\n" + "="*70)
    print(f" RANDOM SAMPLING - {country}")
    print("="*70)
    print(f"Checking {num_samples} random images...")
    print(f"Visualizations will be saved to: {output_dir}\n")
    
    saved_files = []
    
    for i in range(num_samples):
        folder_idx = random.randint(0, len(json_files) - 1)
        
        folder_name = json_files[folder_idx].stem
        img_folder = base_path / "images" / country / folder_name
        img_files = list(img_folder.glob("*.jpg")) + list(img_folder.glob("*.png"))
        
        if img_files:
            img_idx = random.randint(0, len(img_files) - 1)
            
            print(f"\n Sample {i+1}/{num_samples}")
            result = check_single_annotation(base_path, country, folder_idx, img_idx, output_dir)
            
            if result:
                saved_files.append(result)
    
    print(f"\n" + "="*70)
    print(f" Random sampling completed!")
    print(f" Saved {len(saved_files)} visualizations to: {output_dir}")
    print("="*70)
    
    return saved_files


def check_specific_images(base_path, country="srb_passport", image_list=None, output_dir=None):
    """
    Check specific images by their indices
    
    Args:
        base_path: Root directory
        country: Country to check
        image_list: List of tuples (folder_idx, img_idx) to check
        output_dir: Where to save output images
    """
    if image_list is None:
        image_list = [(0, 0), (0, 1), (1, 0)]  # Default: first 2 from folder 0, first from folder 1
    
    base_path = Path(base_path)
    
    if output_dir is None:
        output_dir = base_path / "annotation_checks"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print(f" CHECKING SPECIFIC IMAGES - {country}")
    print("="*70)
    print(f"Checking {len(image_list)} specific images...")
    print(f"Visualizations will be saved to: {output_dir}\n")
    
    saved_files = []
    
    for i, (folder_idx, img_idx) in enumerate(image_list, 1):
        print(f"\n Image {i}/{len(image_list)}: Folder {folder_idx}, Image {img_idx}")
        result = check_single_annotation(base_path, country, folder_idx, img_idx, output_dir)
        
        if result:
            saved_files.append(result)
    
    print(f"\n" + "="*70)
    print(f" Specific check completed!")
    print(f" Saved {len(saved_files)} visualizations to: {output_dir}")
    print("="*70)
    
    return saved_files


if __name__ == "__main__":
    import sys
    
    # Default path
    BASE_PATH = r"C:\Users\Republic Of Computer\Documents\document id"
    
    if len(sys.argv) > 1:
        BASE_PATH = sys.argv[1]
    
    print("\n" + "="*70)
    print(" PASSPORT ANNOTATION VERIFICATION TOOL (FILE OUTPUT VERSION)")
    print("="*70)
    print("All visualizations will be saved as PNG files")
    print()
    print("Choose an option:")
    print()
    print("  [1] Quick check Serbian passports (first 3 images from 3 folders)")
    print("  [2] Check random samples (default: 5 random images)")
    print("  [3] Check specific images (by folder and image indices)")
    print("  [4] Custom quick check (specify number of folders and images)")
    print()
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        quick_check_serbian(BASE_PATH)
    elif choice == "2":
        country = input("Enter country [default: srb_passport]: ").strip() or "srb_passport"
        num_samples = int(input("Number of random samples [default: 5]: ") or "5")
        check_random_samples(BASE_PATH, country, num_samples)
    elif choice == "3":
        country = input("Enter country [default: srb_passport]: ").strip() or "srb_passport"
        print("\nEnter image indices as: folder_idx,img_idx (one per line, empty line to finish)")
        image_list = []
        while True:
            line = input(f"Image {len(image_list)+1} (folder,image): ").strip()
            if not line:
                break
            try:
                folder_idx, img_idx = map(int, line.split(','))
                image_list.append((folder_idx, img_idx))
            except ValueError:
                print("Invalid format. Use: folder_idx,img_idx")
        
        if image_list:
            check_specific_images(BASE_PATH, country, image_list)
        else:
            print("No images specified")
    elif choice == "4":
        num_folders = int(input("Number of folders to check [default: 3]: ") or "3")
        num_images = int(input("Number of images per folder [default: 3]: ") or "3")
        quick_check_serbian(BASE_PATH, num_folders, num_images)
    else:
        print("Invalid choice")
