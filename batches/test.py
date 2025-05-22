import torch
print("Hello World!")
if torch.cuda.is_available():
        
        print(torch.cuda.device_count())
        print(torch.cuda.get_device_name("cuda"))