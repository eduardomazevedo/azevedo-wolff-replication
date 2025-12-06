import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

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

# Convert RGB to hex for seaborn
def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

# Convert RGB to normalized (0-1) for matplotlib
def rgb_to_norm(rgb):
    return tuple(c/255.0 for c in rgb)

# Create figure with multiple subplots
fig = plt.figure(figsize=(14, 15))
gs = fig.add_gridspec(13, 1, hspace=0.6, wspace=0.3)

# Function to plot a palette
def plot_palette(ax, palette, title, n_colors=256):
    if hasattr(palette, '__call__'):
        # If it's a colormap, sample it
        colors_list = [palette(i) for i in np.linspace(0, 1, n_colors)]
    else:
        colors_list = palette
    
    # Create horizontal color bar
    for i, color in enumerate(colors_list):
        rect = patches.Rectangle((i/n_colors, 0), 1/n_colors, 1, 
                                facecolor=color, edgecolor='none')
        ax.add_patch(rect)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.text(0.5, -0.2, title, ha='center', va='top', fontsize=11, fontweight='bold',
            transform=ax.transAxes)

# 1. Dark palettes (good for white paper - light to dark)
ax1 = fig.add_subplot(gs[0, 0])
palette1 = sns.dark_palette(rgb_to_hex(colors['ghibli_palette_warm_red']), 
                            n_colors=256, as_cmap=True, reverse=False)
plot_palette(ax1, palette1, 'Dark Palette: Warm Red → Dark (Light → Dark)')

ax2 = fig.add_subplot(gs[1, 0])
palette2 = sns.dark_palette(rgb_to_hex(colors['ghibli_palette_red']), 
                            n_colors=256, as_cmap=True, reverse=False)
plot_palette(ax2, palette2, 'Dark Palette: Red → Dark (Light → Dark)')

ax3 = fig.add_subplot(gs[2, 0])
palette3 = sns.dark_palette(rgb_to_hex(colors['ghibli_palette_deep_teal']), 
                            n_colors=256, as_cmap=True, reverse=False)
plot_palette(ax3, palette3, 'Dark Palette: Deep Teal → Dark (Light → Dark)')

ax4 = fig.add_subplot(gs[3, 0])
palette4 = sns.dark_palette(rgb_to_hex(colors['ghibli_palette_blue']), 
                            n_colors=256, as_cmap=True, reverse=False)
plot_palette(ax4, palette4, 'Dark Palette: Blue → Dark (Light → Dark)')

# 2. Custom diverging palettes with dark in middle
teal_norm = rgb_to_norm(colors['ghibli_palette_deep_teal'])
blue_norm = rgb_to_norm(colors['ghibli_palette_blue'])
red_norm = rgb_to_norm(colors['ghibli_palette_red'])
warm_red_norm = rgb_to_norm(colors['ghibli_palette_warm_red'])
dark_mid = (0.15, 0.15, 0.15)  # Dark gray for middle

# Diverging: Teal to Warm Red
ax5 = fig.add_subplot(gs[4, 0])
custom_diverging_teal_warm_red = LinearSegmentedColormap.from_list(
    'teal_warm_red',
    [teal_norm, dark_mid, warm_red_norm],
    N=256
)
plot_palette(ax5, custom_diverging_teal_warm_red, 'Diverging: Deep Teal → Dark → Warm Red')

# Diverging: Blue to Red
ax6 = fig.add_subplot(gs[5, 0])
custom_diverging_blue_red = LinearSegmentedColormap.from_list(
    'blue_red',
    [blue_norm, dark_mid, red_norm],
    N=256
)
plot_palette(ax6, custom_diverging_blue_red, 'Diverging: Blue → Dark → Red')

# Diverging: Blue to Warm Red
ax7 = fig.add_subplot(gs[6, 0])
custom_diverging_blue_warm_red = LinearSegmentedColormap.from_list(
    'blue_warm_red',
    [blue_norm, dark_mid, warm_red_norm],
    N=256
)
plot_palette(ax7, custom_diverging_blue_warm_red, 'Diverging: Blue → Dark → Warm Red')

# Diverging: Teal to Red
ax8 = fig.add_subplot(gs[7, 0])
custom_diverging_teal_red = LinearSegmentedColormap.from_list(
    'teal_red',
    [teal_norm, dark_mid, red_norm],
    N=256
)
plot_palette(ax8, custom_diverging_teal_red, 'Diverging: Deep Teal → Dark → Red')

# 3. Sequential palettes (without dark middle): Blue to Red
ax9 = fig.add_subplot(gs[8, 0])
custom_sequential_blue_red = LinearSegmentedColormap.from_list(
    'sequential_blue_red',
    [blue_norm, red_norm],
    N=256
)
plot_palette(ax9, custom_sequential_blue_red, 'Sequential: Blue → Red')

# Sequential: Blue to Warm Red
ax10 = fig.add_subplot(gs[9, 0])
custom_sequential_blue_warm_red = LinearSegmentedColormap.from_list(
    'sequential_blue_warm_red',
    [blue_norm, warm_red_norm],
    N=256
)
plot_palette(ax10, custom_sequential_blue_warm_red, 'Sequential: Blue → Warm Red')

# Sequential: Teal to Red
ax11 = fig.add_subplot(gs[10, 0])
custom_sequential_teal_red = LinearSegmentedColormap.from_list(
    'sequential_teal_red',
    [teal_norm, red_norm],
    N=256
)
plot_palette(ax11, custom_sequential_teal_red, 'Sequential: Deep Teal → Red')

# Sequential: Teal to Warm Red
ax12 = fig.add_subplot(gs[11, 0])
custom_sequential_teal_warm_red = LinearSegmentedColormap.from_list(
    'sequential_teal_warm_red',
    [teal_norm, warm_red_norm],
    N=256
)
plot_palette(ax12, custom_sequential_teal_warm_red, 'Sequential: Deep Teal → Warm Red')

plt.suptitle('Sequential Color Palettes from Ghibli Colors\n(Optimized for White Paper)', 
             fontsize=14, fontweight='bold', y=0.995)
plt.savefig('palette_samples.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Sequential palettes saved to palette_samples.png")
