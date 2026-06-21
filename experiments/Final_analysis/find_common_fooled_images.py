import pandas as pd
import os

# Define file paths
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../results/Final_analysis"))
critical_region_file = os.path.join(base_dir, "Critical_region_attack.csv")
full_image_file = os.path.join(base_dir, "Full_image_attack.csv")
random_mask_file = os.path.join(base_dir, "random_mask_attack_log.csv")

# Load data
df_critical = pd.read_csv(critical_region_file)
df_full = pd.read_csv(full_image_file)
df_random = pd.read_csv(random_mask_file)

# Function to extract a set of images where fooled is true
def get_fooled_images(df):
    # Handle boolean or string representation of True
    return set(df[ (df['fooled'] == True) | (df['fooled'] == 'True') ]['image'])

# Get fooled images for each method
critical_fooled = get_fooled_images(df_critical)
full_fooled = get_fooled_images(df_full)
random_fooled = get_fooled_images(df_random)

# Find common fooled images in all three methods
common_images = critical_fooled.intersection(full_fooled).intersection(random_fooled)

# Convert to sorted list for display
common_images_list = sorted(list(common_images))

print(f"Number of common images fooled across all 3 methods: {len(common_images_list)}")
if common_images_list:
    print("Common Fooled Images:")
    for img in common_images_list:
        print(f" - {img}")

# Optionally save the list to a text file
output_file = os.path.join(base_dir, "common_fooled_images.txt")
with open(output_file, 'w') as f:
    f.write(f"Number of common images fooled across all 3 methods: {len(common_images_list)}\n\n")
    for img in common_images_list:
        f.write(f"{img}\n")

print(f"\nSaved the list to: {output_file}")
