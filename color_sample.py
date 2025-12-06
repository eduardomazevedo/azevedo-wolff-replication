import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Define the Ghibli color palette (RGB values 0-255)
colors = {
    'ghibli_palette_red': (255, 92, 92),
    'ghibli_palette_gold': (255, 215, 0),
    'ghibli_palette_deep_teal': (59, 105, 120),
    'ghibli_palette_cream': (249, 245, 227),
    'ghibli_palette_light_gray': (218, 218, 218),
    'ghibli_palette_gray': (108, 122, 142),
    'ghibli_palette_blue': (143, 177, 233),
    'ghibli_palette_green': (166, 215, 132),
    'ghibli_palette_muted_green': (132, 165, 157),
    'ghibli_palette_warm_red': (242, 132, 130),
}

# Convert RGB values from 0-255 to 0-1 range for matplotlib
colors_normalized = {name: tuple(c/255.0 for c in rgb) for name, rgb in colors.items()}

# Create figure
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, len(colors))
ax.axis('off')

# Create color swatches
for i, (name, color) in enumerate(colors_normalized.items()):
    # Draw color rectangle
    rect = patches.Rectangle((0, i), 3, 0.8, linewidth=1, 
                            edgecolor='black', facecolor=color)
    ax.add_patch(rect)
    
    # Add color name
    ax.text(3.2, i + 0.4, name.replace('ghibli_palette_', '').replace('_', ' ').title(),
            va='center', fontsize=11, fontweight='bold')
    
    # Add RGB values
    rgb_values = colors[name]
    ax.text(3.2, i + 0.15, f'RGB({rgb_values[0]}, {rgb_values[1]}, {rgb_values[2]})',
            va='center', fontsize=9, style='italic', color='gray')

plt.tight_layout()
plt.savefig('color_sample.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Color palette saved to color_sample.png")
