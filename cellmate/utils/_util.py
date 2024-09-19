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
import pickle


def hash_func(data):
    return {k: i for i, k in enumerate(data, 0)}


def dump_to_pkl(obj: object, filename: str):
    """Save object as pickle
    obj: object
    filename: save path
    """
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)


def load_from_pkl(filename: str):
    """Load pickle as object
    filename: str, file path
    """
    with open(filename, 'rb') as file:
        obj = pickle.load(file)
    return obj
