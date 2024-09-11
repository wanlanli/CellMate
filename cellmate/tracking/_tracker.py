# def cell_tracking(image, method="", generation=True, *args, **kwargs):
#     """apply tracking based on different method.

#     method: "iou, perfet, sort, kalman, optical flow"
#     """
#     if method == "iou":
#         from tracking._iou_tracker import Tracker
#     elif method == "hausdorf":
#         from tracking._hausdorf_tracker import Tracker
#     trace = Tracker(image, *args, **kwargs)
#     _ = trace()
#     pass
