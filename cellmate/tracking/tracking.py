# from typing import Tuple, Union

# import numpy as np

# from cellmating.sort import Tracker


# def tracking(image: np.ndarray, channel: int = 3, is_save_tracker: bool = False,
#              *args, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, object]]:
#     """
#     Tracks cells across different frames in a time-lapse movie (image) to assign unique IDs to each cell over time.
#     The image is expected to be a 4D numpy array representing a sequence of frames over time, each containing multiple
#     channels of data, including one for segmentation results. This function uses the segmentation channel to track cells
#     across the frames, appending the tracking results as an additional channel to the output image.

#     Parameters:
#     ----------
#     image : np.ndarray
#         A 4D numpy array with shape (Time, Channel, Width, Height), representing a time-lapse movie where each frame
#         contains multiple channels of data, including at least one segmentation channel.

#     channel : int, optional
#         The index (0-based) of the channel within `image` that contains the segmentation results. Defaults to 3,
#         indicating that the fourth channel is the segmentation channel.

#     is_save_tracker : bool
#         If True, the function returns both the tracked image array and the tracker object. If False, only the tracked
#         image array is returned.

#     *args, **kwargs :
#         Additional arguments and keyword arguments for future use or for passing optional parameters to underlying
#         methods or functions involved in the tracking process.

#     Returns:
#     ---------
#     If `is_save_tracker` is False:
#         tracked : np.ndarray
#             A 4D numpy array with shape (Time, Channel + 1, Width, Height), with tracking results appended as an
#             additional channel. Each cell tracked over time is assigned a unique ID.
#     If `is_save_tracker` is True:
#         (tracked, tracer) : Tuple[np.ndarray, object]
#             - tracked: As described above.
#             - tracer: The tracker object, which includes the tracking network and relationships like cell division
#               and fusion.

#     Note:
#     -----
#     The implementation of this function depends on cellmating.sort.Tracker for cell tracking based on segmentation results.
#     The exact method are not described here.
#     """
#     if image.ndim == 3:
#          tracker = Tracker(image, *args, **kwargs)
#     else:
#         tracker = Tracker(image[:, channel], *args, **kwargs)
#     _ = tracker()
#     if image.ndim == 3:
#         tracked = tracker.to_image()
#     else:
#         tracked = np.zeros((image.shape[0], image.shape[1]+1, image.shape[2], image.shape[3]), 
#                            dtype=image.dtype)
#         tracked[:, 0:image.shape[1]] = image[:, 0:image.shape[1]]
#         tracked[:, image.shape[1]] = tracker.to_image()
#     if is_save_tracker:
#         return tracked, tracker
#     else:
#         return tracked


# if __name__ == '__main__':
#     pass
