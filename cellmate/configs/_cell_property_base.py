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

class _PropertyBase:
    def __init__(self) -> None:
        pass

    def set_attr_items(self, obj):
        for k, v in obj.__dict__.items():
            if not k.startswith("_"):
                setattr(self, k, v)

    def get_attr_items(self):
        data = {}
        for k, v in self.__dict__.items():
            if not k.startswith("_"):
                data[k] = v
        return data


def hash_func(data):
    return {k: i for i, k in enumerate(data, 0)}


def get_attr_items(obj):
    data = {}
    for k, v in obj.__dict__.items():
        if not k.startswith("_"):
            data[k] = v
    return data
