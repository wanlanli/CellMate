# Copyright 2024 wlli
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Constant used to divide the label to extract semantic and instance labels.
DIVISION = 1000

# The segmentation model generates a label that encodes both a semantic label and an instance label.
# The semantic label represents the class or type of the object (e.g., different cell types or structures).
# The instance label distinguishes between different instances of the same semantic type (e.g., multiple cells of the same type).
# Example: If label = 1234567, then semantic_label = label // DIVISION = 1234567 // 1000 = 1234.
#                              then instance_label = label % DIVISION = 1234567 % 1000 = 567.

IMAGE_CONTOURS_LENGTH = 60
TRACK_FEATURE_DIMENSION = 8
