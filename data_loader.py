import torchvision
import torch
from torchvision import datasets,transforms

def load_data(root_dir,domain,batch_size):
    transform = transforms.Compose([               #Compose的主要作用是将多个变换组合在一起
        transforms.Grayscale(),       #Grayscale的作用是将图像转换为灰度图像，默认通道数为1
        transforms.Resize([28, 28]),  #Resize的作用是对图像进行缩放
        transforms.ToTensor(), #ToTensor的作用是将PIL Image或numpy.ndarray转为pytorch的Tensor，并会将像素值由[0, 255]变为[0, 1]之间

        transforms.Normalize(mean=(0,),std=(1,)),])
    image_folder = datasets.ImageFolder(
            root=root_dir + domain,
            transform=transform)
    # 从数据库中每次抽出batch size个样本
    #torch.utils.data.DataLoader把训练数据分成多个小组，此函数每次抛出一组数据。直至把所有的数据都抛出
    data_loader = torch.utils.data.DataLoader(dataset=image_folder,batch_size=batch_size,shuffle=True,num_workers=2,drop_last=True)  #出现batch_size小于预期的情况指定drop_last = True解决)
    return data_loader

def load_test(root_dir,domain,batch_size):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize([28, 28]),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0,), std=(1,)),])
    image_folder = datasets.ImageFolder(
        root=root_dir + domain,
        transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset=image_folder, batch_size=batch_size, shuffle=False, num_workers=2 ,drop_last=True)
    return data_loader