from pathlib import Path

import numpy as np


def save_histogram_analysis(original: np.ndarray, filtered: np.ndarray, output_path: Path) -> None:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    original_clipped = np.clip(np.asarray(original, dtype=np.float64), 0.0, 1.0)
    filtered_clipped = np.clip(np.asarray(filtered, dtype=np.float64), 0.0, 1.0)
    diff_image = np.abs(filtered_clipped - original_clipped)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].imshow(original_clipped, cmap='gray', vmin=0.0, vmax=1.0)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(filtered_clipped, cmap='gray', vmin=0.0, vmax=1.0)
    axes[0, 1].set_title('Filtered')
    axes[0, 1].axis('off')

    diff_plot = axes[1, 0].imshow(diff_image, cmap='magma')
    axes[1, 0].set_title('Absolute Difference')
    axes[1, 0].axis('off')
    fig.colorbar(diff_plot, ax=axes[1, 0], fraction=0.046, pad=0.04)

    axes[1, 1].hist(
        original_clipped.ravel(),
        bins=256,
        range=(0.0, 1.0),
        density=True,
        alpha=0.55,
        color='steelblue',
        label='Original',
    )
    axes[1, 1].hist(
        filtered_clipped.ravel(),
        bins=256,
        range=(0.0, 1.0),
        density=True,
        alpha=0.55,
        color='darkorange',
        label='Filtered',
    )
    axes[1, 1].set_title('Histogram Comparison')
    axes[1, 1].set_xlabel('Intensity')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
