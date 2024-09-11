# Image:
CONTOURS_LENGTH = 60
SKELETON_LENGTH = 25
RESOLUTION = 1
# Segment:
# # The segmentation model generates a label that encodes both a semantic label and an instance label.
# # The semantic label represents the class or type of the object (e.g., different cell types or structures).
# # The instance label distinguishes between different instances of the same semantic type (e.g., multiple cells of the same type).
# # Example: If label = 1234567, then semantic_label = label // DIVISION = 1234567 // 1000 = 1234.
# #                              then instance_label = label % DIVISION = 1234567 % 1000 = 567.

DIVISION = 1000
# Track:
FEATURE_DIMENSION = 8
