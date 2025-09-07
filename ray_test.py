import os
os.environ["HIP_VISIBLE_DEVICES"] = os.environ.get("ROCR_VISIBLE_DEVICES", "")
os.environ.pop("ROCR_VISIBLE_DEVICES", None)

import ray
from ray import tune

ray.init()

@ray.remote(num_gpus=1)
def train():
    import torch
    print("GPU available?", torch.cuda.is_available())
    print("Device:", torch.cuda.get_device_name(0))
    return 42

print(ray.get(train.remote()))
