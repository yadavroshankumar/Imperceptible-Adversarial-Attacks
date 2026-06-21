import pandas as pd
import matplotlib.pyplot as plt
import os

# Define file paths
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../results/Final_analysis"))
critical_region_file = os.path.join(base_dir, "Critical_region_attack.csv")
full_image_file = os.path.join(base_dir, "Full_image_attack.csv")
random_mask_file = os.path.join(base_dir, "random_mask_attack_log.csv")
agent_attack_file = os.path.join(base_dir, "agent_attack_log.csv")

# Load data
df_critical = pd.read_csv(critical_region_file)
df_full = pd.read_csv(full_image_file)
df_random = pd.read_csv(random_mask_file)
df_agent = pd.read_csv(agent_attack_file)

# Extract fooled == True count handling both boolean and string representations
count_critical = (df_critical['fooled'] == True).sum() + (df_critical['fooled'] == 'True').sum()
count_full = (df_full['fooled'] == True).sum() + (df_full['fooled'] == 'True').sum()
count_random = (df_random['fooled'] == True).sum() + (df_random['fooled'] == 'True').sum()
count_agent = (df_agent['fooled'] == True).sum() + (df_agent['fooled'] == 'True').sum()

print("Successful Attacks:")
print(f"Full Image Attack: {count_full}")
print(f"Random Mask Attack: {count_random}")
print(f"Critical Region Attack: {count_critical}")
print(f"Agent Attack: {count_agent}")

# Create bar chart
labels = ['Full Image', 'Random Mask', 'Critical Region', 'Agent Attack']
counts = [count_full, count_random, count_critical, count_agent]

plt.figure(figsize=(10, 6))
bars = plt.bar(labels, counts, color=['#55A868', '#C44E52', '#4C72B0', '#8172B3'])

plt.ylabel('Number of Successful Attacks')
plt.title('Comparison of Adversarial Attack Success (All Methods)')
plt.ylim(0, max(counts) + sum(counts)*0.1 + 5) # Adding top margin

# Add data labels on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, int(yval), va='bottom', ha='center', fontweight='bold')

plt.tight_layout()

# Save the plot
output_path = os.path.join(base_dir, "all_attack_success_comparison.png")
plt.savefig(output_path, dpi=300)
print(f"\nBar graph successfully saved to: {output_path}")

