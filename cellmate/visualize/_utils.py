import numpy as np
from ._colormap import COLORMAP


def plot_cell_types(cellnet, frame=0, strain_colors=None, ax=None):
    """Overlay each cell's predicted `strain_type` (set by
    `CellNetwork.create_cell_type`) on one frame of `cellnet.image`, labeled
    with its id -- for visually spot-checking the classification.

    Parameters
    ----------
    cellnet: CellNetwork
        Must already have had `create_cell_type` called on it.
    frame: int
        Which frame of `cellnet.image` to draw.
    strain_colors: dict[int, str], optional
        strain_type -> matplotlib color. Defaults to
        {0: "lightgray", 1: "tab:red", 2: "tab:blue", 3: "tab:purple"}
        (0 = unclassified, 1 = ch0 marker/h+, 2 = ch1 marker/h-, 3 = both).
    ax: matplotlib.axes.Axes, optional
        Axes to draw into; a new figure/axes is created if omitted.

    Returns
    -------
    ax: matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    if strain_colors is None:
        strain_colors = {
            0: (211, 211, 211),  # light gray
            1: (0, 255, 0),      # green
            2: (255, 0, 0),      # red
            3: (255, 165, 0),    # orange
        }

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    mask = cellnet.image[frame]
    ax.imshow(mask > 0, cmap="gray", alpha=0.3)

    colored_mask = np.full((*mask.shape, 3), 255, dtype=np.uint8)
    id_by_label = {v: k for k, v in cellnet.label_map[frame].items()}
    for cell_label in np.unique(mask):
        if cell_label == 0:
            continue
        cell_id = id_by_label.get(cell_label)
        if cell_id is None or cell_id not in cellnet.cells:
            continue
        strain_type = cellnet.cells[cell_id].strain_type
        if strain_type in strain_colors:
            colored_mask[mask == cell_label] = strain_colors[strain_type]
        # mask = labels == cell_label
        # rows, cols = np.nonzero(mask)
        # overlay = np.zeros((*labels.shape, 4))
        # overlay[mask] = to_rgba(color, alpha=0.6)
        # ax.imshow(overlay)
        # ax.text(cols.mean(), rows.mean(), f"{cell_id}\n({strain_type})",
        #         color="white", ha="center", va="center", fontsize=8)
    ax.imshow(colored_mask)
    ax.set_title(f"frame {frame} -- predicted strain_type")
    ax.axis("off")
    return ax


def label2rgb(img, colormap=None):
    """
    Convert a labeled image to an RGB image using a colormap.

    Parameters:
    -----------
    img : np.array
        Labeled image array.
    colormap : matplotlib.colors.ListedColormap, optional
        Colormap to use for mapping labels to colors. If None, a default colormap is used.

    Returns:
    --------
    data : np.array
        RGB image array with shape (height, width, 3) and dtype uint8.
    """
    try:
        mapped_labels_flat = np.unique(img)
        index = np.arange(0, len(mapped_labels_flat))
        if colormap is None:
            colormap = COLORMAP['default'].resampled(len(index))

        label_index_map = {}
        f = 0
        for i in range(0, mapped_labels_flat.max()+1):
            if i == mapped_labels_flat[f]:
                label_index_map[i] = f
                f += 1
            else:
                label_index_map[i] = 0

        label_to_color = np.stack([colormap(label_index_map[i])[0:3] for i in np.arange(mapped_labels_flat.max()+1)])

        def __func(x):
            return label_to_color[x]

        data = (__func(img)*255).astype(np.uint8)
        return data
    except Exception as err:
        print(f"Unexpected error: {err}, {type(err)}")
