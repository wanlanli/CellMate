import numpy as np
from ._colormap import COLORMAP


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
