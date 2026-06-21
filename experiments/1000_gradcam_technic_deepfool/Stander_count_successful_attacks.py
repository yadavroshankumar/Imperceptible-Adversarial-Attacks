import csv
import os

LOG_FILE = "../../results/1000_gradcam_technic_deepfool/standard_attack_log.csv"

def main():
    if not os.path.exists(LOG_FILE):
        print(f"Error: Log file '{LOG_FILE}' not found.")
        return

    sucessful_attacks = 0
    fast_successful_attacks = 0
    total_images = 0

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_images += 1
            
            # Check if fooled
            fooled = row.get("fooled", "False").strip()
            
            # Extract SSIM and Iterations
            ssim_str = row.get("final_SSIM", "0").strip()
            iters_str = row.get("total_iterations", "0").strip()
            
            # If the script failed on an image, it outputs 'N/A'
            if ssim_str == "N/A" or iters_str == "N/A":
                continue
                
            try:
                ssim = float(ssim_str)
                iters = int(iters_str)
            except ValueError:
                continue
                
            # Check primary conditions (Fooled & SSIM >= 0.90)
            if fooled == "True" and ssim >= 0.90:
                sucessful_attacks += 1
                
                # Check secondary condition: under 100 iterations
                if iters < 100:
                    fast_successful_attacks += 1

    print(f"Total processed images: {total_images}")
    print(f"Successfully fooled images with SSIM >= 0.90: {sucessful_attacks}")
    print(f"Successfully fooled (SSIM >= 0.90) in < 100 iterations: {fast_successful_attacks}")
    
    if total_images > 0:
        success_rate = (sucessful_attacks / total_images) * 100
        fast_success_rate = (fast_successful_attacks / total_images) * 100
        print(f"Overall Success Rate: {success_rate:.2f}%")
        print(f"Fast Success Rate (< 100 iters): {fast_success_rate:.2f}%")

if __name__ == "__main__":
    main()
