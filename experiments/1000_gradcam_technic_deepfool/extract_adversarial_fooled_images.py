import csv
import os
import shutil

csv_path = '../../results/1000_gradcam_technic_deepfool/agent_attack_log.csv'
source_dir = '../../results/1000_gradcam_technic_deepfool/Adversarial_Dog_AIAgent'
dest_dir = 'sucessfull_fooled_image'

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

copied_count = 0
with open(csv_path, 'r', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get('fooled') == 'True':
            image_name = row.get('image')
            if not image_name:
                continue
            
            # Determine the adversarial image filename
            base_name, _ = os.path.splitext(image_name)
            adv_image_name = f"{base_name}_agent_adv.png"
            
            source_path = os.path.join(source_dir, adv_image_name)
            dest_path = os.path.join(dest_dir, adv_image_name)
            
            if os.path.exists(source_path):
                shutil.copy(source_path, dest_path)
                copied_count += 1
            else:
                print(f"File not found: {source_path}")

print(f"Successfully copied {copied_count} adversarial fooled images.")
