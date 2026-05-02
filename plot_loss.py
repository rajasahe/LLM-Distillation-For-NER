import json
import matplotlib.pyplot as plt
import os

with open('training_loss_log.json', 'r') as f:
    data = json.load(f)

steps = []
ce_losses = []
kl_losses = []
total_losses = []

for row in data:
    if row.get('epoch') == 1:
        steps.append(row['step'])
        ce_losses.append(row.get('ce_loss', 0))
        kl_losses.append(row.get('kl_loss', 0))
        total_losses.append(row.get('loss', 0))

fig, axs = plt.subplots(1, 3, figsize=(8, 3))

# Total Loss Subplot
axs[0].plot(steps, total_losses, label='Total Loss', color='red', alpha=0.9, linewidth=1.5)
axs[0].set_xlabel('Step')
axs[0].set_ylabel('Loss')
axs[0].set_title('Total Loss')
axs[0].grid(True, linestyle='--', alpha=0.6)

# KL Divergence Subplot
axs[1].plot(steps, kl_losses, label='KL Divergence', color='green', alpha=0.8, linewidth=1.5)
axs[1].set_xlabel('Step')
axs[1].set_title('KL Divergence Loss')
axs[1].grid(True, linestyle='--', alpha=0.6)

# Cross Entropy Subplot
axs[2].plot(steps, ce_losses, label='Cross Entropy', color='blue', alpha=0.8, linewidth=1.5)
axs[2].set_xlabel('Step')
axs[2].set_title('Cross Entropy Loss')
axs[2].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()

output_path = r'C:\Users\Raja\Coriolis\slm-distillation\loss_plot.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved plot to {output_path}")
