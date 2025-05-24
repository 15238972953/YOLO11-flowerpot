import torch
import torchvision
print(torch.cuda.is_available())
print(torch.backends.cudnn.is_available())
print(torch.version.cuda)
print(torch.backends.cudnn.version())

print(torch.__version__)
print(torchvision.__version__)