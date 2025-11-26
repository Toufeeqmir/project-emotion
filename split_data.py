import os
import shutil
import random
import sys

# --- Configuration ---

# This is your original folder with all 7 emotion subfolders
source_dir = r'E:\python\dataset'

# This is where the new 'train', 'validation', and 'test' folders will be created
output_dir = r'E:\python\split_data'

# Define the split ratios
train_ratio = 0.70
val_ratio = 0.15
# test_ratio will be whatever is left over (e.g., 0.15)

# --- End of Configuration ---


def split_data():
    print(f"Reading data from: {source_dir}")
    print(f"Saving splits to: {output_dir}")
    print("-" * 30)

    # 1. Create the new folder structure (train/angry, validation/angry, etc.)
    print("Creating new directory structure...")
    
    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"Error: Source directory not found at {source_dir}")
        print("Please check the 'source_dir' path in your script.")
        return

    try:
        # Get list of class names (e.g., 'angry', 'happy')
        class_names = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
        if not class_names:
            print(f"Error: No class folders (like 'angry', 'happy') found in {source_dir}")
            return
            
        print(f"Found classes: {', '.join(class_names)}")

        for split in ['train', 'validation', 'test']:
            split_path = os.path.join(output_dir, split)
            os.makedirs(split_path, exist_ok=True)
            for class_name in class_names:
                class_path = os.path.join(split_path, class_name)
                os.makedirs(class_path, exist_ok=True)
        print("Directory structure created successfully.")

    except Exception as e:
        print(f"Error creating directories: {e}")
        return

    # 2. Go through each class, shuffle, and move files
    print("\nStarting to move files...")
    
    for class_name in class_names:
        print(f"  Processing class: {class_name}")
        
        class_source_path = os.path.join(source_dir, class_name)
        
        # Get all image files
        try:
            image_files = [f for f in os.listdir(class_source_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
        except FileNotFoundError:
            print(f"    Warning: Folder not found or empty: {class_source_path}")
            continue

        if not image_files:
            print(f"    Warning: No image files found for class '{class_name}'")
            continue
            
        # Shuffle them randomly to ensure a fair split
        random.shuffle(image_files)
        
        # Calculate split sizes
        total_images = len(image_files)
        train_count = int(total_images * train_ratio)
        val_count = int(total_images * val_ratio)
        
        # --- START OF FIX ---
        # These lines were missing before
        train_files = image_files[:train_count]
        val_files = image_files[train_count : train_count + val_count]
        # --- END OF FIX ---
        
        test_files = image_files[train_count + val_count :] # The rest go to test
        
        # Function to move files and handle errors
        def move_files(files, split_name):
            moved_count = 0
            for f in files:
                src = os.path.join(class_source_path, f)
                dst = os.path.join(output_dir, split_name, class_name, f)
                try:
                    shutil.move(src, dst)
                    moved_count += 1
                except Exception as e:
                    print(f"    Error moving file {src} to {dst}: {e}")
            return moved_count

        # Move the files to their new homes
        tr_moved = move_files(train_files, 'train')
        va_moved = move_files(val_files, 'validation')
        te_moved = move_files(test_files, 'test')
        
        print(f"    Moved {tr_moved} to train, {va_moved} to validation, {te_moved} to test.")

    print("\n" + "=" * 30)
    print("--- ALL FILES HAVE BEEN MOVED ---")
    print(f"Your original '{source_dir}' folder should now be empty.")
    print(f"Your new data is ready in '{output_dir}'.")
    print("=" * 30)

# This line makes the script run when you call it from the terminal
if __name__ == "__main__":
    split_data()