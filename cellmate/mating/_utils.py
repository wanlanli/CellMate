# import numpy as np
# import pandas as pd


# def background(fluorescent_image, masks, threshold: float = 10) -> np.array:
#     """
#     Get the background threshold for every channel.

#     Parameters:
#     ----------
#     threshold: float
#         The percentile of data not taken into account.

#     Returns:
#     -------
#     bg_threshold: np.array with shape [frame x number of channels]
#         An array containing background thresholds for each frame and channel.
#     """
#     masked = fluorescent_image*((masks == 0)[:, None, :, :])
#     channel_number = fluorescent_image.shape[1]
#     bg_threshold = np.zeros((fluorescent_image.shape[0],  channel_number))
#     for i in range(0, channel_number):
#         for f in range(0, fluorescent_image.shape[0]):
#             value = flatten_nonzero_value(masked[f, i])
#             if value.sum() == 0:
#                 bg_threshold[f, i] = 0
#             else:
#                 floor = np.percentile(value, threshold)
#                 celling = np.percentile(value, 100-threshold)
#                 value = value[(value >= floor) & (value <= celling)]
#                 bg_threshold[f, i] = np.mean(value)
#     return bg_threshold


# def instance_fluorescent_intensity(masks,  fluorescent_image, bg=None, measure_line=70):
#     data_sheet = []
#     label_list = np.unique(masks)[1:]
#     for label in label_list:
#         mask = masks == label
#         index = mask.sum(axis=(1, 2))
#         index = np.where(index)[0]
#         intensity = mask[:, None, :, :] * fluorescent_image

#         for f in index:
#             data = [label, f]
#             for ch in range(0, fluorescent_image.shape[1]):
#                 ch_v = np.percentile(flatten_nonzero_value(intensity[f][ch]), measure_line)
#                 data.append(ch_v)
#                 if bg is not None:
#                     data.append(bg[f, ch])
#             data_sheet.append(data)
#     data_sheet = pd.DataFrame(data_sheet)
#     column = ["label", "frame"]
#     if bg is not None:
#         for i in range(0, fluorescent_image.shape[1]):
#             column += [f"ch_{i}", f"bg_{i}"]
#     else:
#         for i in range(0, fluorescent_image.shape[1]):
#             column.append(f"ch_{i}")
#     data_sheet.columns = column
#     return data_sheet


# def flatten_nonzero_value(data):
#     """flatten all non-zero values in data
#     data: array_like
#     """
#     flatten = data.flatten()
#     flatten = flatten[flatten > 0]
#     return flatten
