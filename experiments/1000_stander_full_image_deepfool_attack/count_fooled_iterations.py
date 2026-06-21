import csv
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Count successful attacks with iterations <= 1000")
    parser.add_argument('--log_file', type=str, default='../../results/1000_stander_full_image_deepfool_attack/agent_attack_log.csv', help='Path to the log file')
    args = parser.parse_args()

    log_path = args.log_file
    if not os.path.exists(log_path):
        print(f"Error: The file '{log_path}' does not exist.")
        return

    count = 0
    total_fooled = 0
    
    with open(log_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            fooled_val = row.get('fooled', '').strip().upper()
            iterations_str = row.get('total_iterations', '0').strip()
            
            if fooled_val == 'TRUE':
                total_fooled += 1
                try:
                    iterations = int(iterations_str)
                    if iterations <= 1000:
                        count += 1
                except ValueError:
                    print(f"Warning: Could not parse total_iterations '{iterations_str}' for image {row.get('image', 'Unknown')}.")

    print(f"Total attacks where fooled == True: {total_fooled}")
    print(f"Number of attacks where fooled == True AND total_iterations <= 1000: {count}")

if __name__ == "__main__":
    main()
