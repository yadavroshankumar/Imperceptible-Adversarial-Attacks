import pandas as pd
import os

# Define the directory
directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../results/Final_analysis"))

# Define the mapping for each file and its relevant columns
files_info = [
    {
        "filename": "../../results/Final_analysis/agent_attack_log.csv",
        "fooled_col": "fooled",
        "iter_col": "total_iterations",
        "ssim_col": "final_SSIM"
    },
    {
        "filename": "../../results/Final_analysis/Full_image_attack.csv",
        "fooled_col": "fooled",
        "iter_col": "iterations",
        "ssim_col": "final_SSIM"
    },
    {
        "filename": "../../results/Final_analysis/Critical_region_attack.csv",
        "fooled_col": "fooled",
        "iter_col": "total_iterations",
        "ssim_col": "final_SSIM"
    },
    {
        "filename": "../../results/Final_analysis/random_mask_attack_log.csv",
        "fooled_col": "fooled",
        "iter_col": "total_iterations",
        "ssim_col": "final_SSIM"
    }
]

print("Mean SSIM and Mean Iterations for Fooled == True\n" + "-"*50)

for info in files_info:
    filepath = os.path.join(directory, info["filename"])
    if not os.path.exists(filepath):
        print(f"File not found: {info['filename']}")
        continue
        
    try:
        # Read the CSV
        df = pd.read_csv(filepath)
        
        # Filter to choose rows where fooled == True
        fooled_df = df[df[info['fooled_col']] == True].copy()
        
        # Ensure the columns are numeric before calculating mean
        fooled_df[info['ssim_col']] = pd.to_numeric(fooled_df[info['ssim_col']], errors='coerce')
        fooled_df[info['iter_col']] = pd.to_numeric(fooled_df[info['iter_col']], errors='coerce')
        
        # Calculate means
        mean_ssim = fooled_df[info['ssim_col']].mean()
        mean_iter = fooled_df[info['iter_col']].mean()
        
        print(f"File: {info['filename']}")
        print(f"  Mean SSIM:       {mean_ssim:.4f}")
        print(f"  Mean Iterations: {mean_iter:.2f}")
        print("-" * 50)
        
    except Exception as e:
        print(f"Error processing {info['filename']}: {e}")
