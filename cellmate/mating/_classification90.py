import numpy as np
from skimage import morphology
import scipy.ndimage as ndi


def fluorescent_intensity_h90_single(fluorescent_image, mask, erosion_k=13, dilation_k=1, conv_k=17):
    closing = morphology.binary_erosion(mask, footprint=morphology.disk(erosion_k))
    opening = morphology.binary_dilation(mask, footprint=morphology.disk(dilation_k))
    cc = ndi.convolve(closing*fluorescent_image/(conv_k*conv_k), np.ones((conv_k, conv_k)), mode='constant', cval=0)
    idx = np.argmax(cc)
    row, col = np.unravel_index(idx, cc.shape)

    mask_edge = ~closing & opening

    edge_list = fluorescent_image[mask_edge]
    edge_intensity = np.mean(edge_list[edge_list > np.percentile(edge_list, 70)])
    core_intensity = cc[row, col]
    bg_intensity = np.percentile(fluorescent_image[closing], 10)

    return edge_intensity, core_intensity, bg_intensity


def instance_fluorescent_intensity_h90(fluorescent_image, masks, *args, **kwargs):
    data_sheet = []
    label_list = np.unique(masks)[1:]
    for label in label_list:
        mask = masks == label
        x, y = np.where(mask)
        x_min = max(0, np.min(x) - 10)
        x_max = min(np.max(x)+10, mask.shape[0])
        y_min = max(0, np.min(y) - 10)
        y_max = min(np.max(y)+10, mask.shape[1])

        mask = mask[x_min:x_max, y_min:y_max]
        f_cropped = fluorescent_image[x_min:x_max, y_min:y_max]
        edge_intensity, core_intensity, bg_intensity = fluorescent_intensity_h90_single(f_cropped, mask, *args, **kwargs)
        data = [label, edge_intensity, core_intensity, bg_intensity]
        data_sheet.append(data)
    return data_sheet
