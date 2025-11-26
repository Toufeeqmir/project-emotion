import os

# Main dataset path
data_dir = r'E:\python\split_data'
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

class_counts = {}
total_images = 0

print("--- Image Counts Per Class ---")

# Walk through all subdirectories (train/val/test and emotion folders)
for root, dirs, files in os.walk(data_dir):
    for file_name in files:
        if file_name.lower().endswith(image_extensions):
            # Get the emotion name (last folder before the file)
            class_name = os.path.basename(os.path.dirname(os.path.join(root, file_name)))
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            total_images += 1

# Print results
for class_name, count in class_counts.items():
    print(f"{class_name}: {count} images")

print("------------------------------")
print(f"Total: {total_images} images")

print("\n--- Recommended Split Sizes (70% Train, 15% Val, 15% Test) ---")
for class_name, count in class_counts.items():
    train_count = int(count * 0.70)
    val_count = int(count * 0.15)
    test_count = count - train_count - val_count
    print(f"[{class_name}] -> Train: {train_count}, Val: {val_count}, Test: {test_count}")
