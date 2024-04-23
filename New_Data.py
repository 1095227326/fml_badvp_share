
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import random
import numpy as np
from copy import deepcopy
import pickle
import os
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
from torch.utils.data import  Dataset, random_split
from tqdm import tqdm
import random
poisondata_idxs_random = random.Random(42) 
rng = np.random.default_rng(seed=42)

def save_image_data(save_path, origin_data, depth=0):
    data_str = 'origin_data'
    for i in range(depth):
        data_str += '.dataset'
    folder_data = eval(data_str)

    with open(save_path,"wb") as f:
        for i in range(len(origin_data)):
            real_idx = i
            data_str = 'origin_data'
            for j in range(depth):
                real_idx = eval(data_str+'.indices[{}]'.format(real_idx))
                data_str += '.dataset'
            
            path, target = folder_data.samples[real_idx]
            sample = folder_data.loader(path)

            pickle.dump((sample, target),f)

    print('Data has been successfully saved to {}!'.format(save_path))

class MyEuroSAT(torch.utils.data.Dataset):
    def __init__(self, file_path, transform) -> None:
        super().__init__()
        self.classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

        self.transform = transform
        self.data, self.targets = [], []

        with open(file_path, "rb") as f:
            while True:
                try:
                    sample, target = pickle.load(f)
                    self.data.append(sample)
                    self.targets.append(target)
                except EOFError as e:
                    break

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, target
        
class CustomDataset(Dataset):
    def __init__(self, data, targets,tags=[], class_names=[], dataset_name='default'):
        self.data = []
        self.targets = []
        self.tags = []
        self.data.extend(list(data))
        self.targets.extend(list(targets))
        
        if len(tags) > 0:
            self.tags.extend(tags)
            
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])
        
        self.class_names = class_names
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        img = self.data[index]
        label = self.targets[index]
        # img = np.array(img, dtype=np.uint8)
        # print(img)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        
        if len(self.tags) > 0:
            tag = self.tags[index]
            return img, label,tag

        return img, label


def get_full_data(dataset_name):
    
    if dataset_name == 'cifar10':

        dataset = datasets.CIFAR10(root='./data/{}'.format(dataset_name), train=True, download=True)
        test_dataset = datasets.CIFAR10(root='./data/{}'.format(dataset_name), train=False,download=True)
        class_names = deepcopy(dataset.classes)
        
        o_train_data = deepcopy(dataset.data)
        o_train_labels = deepcopy(dataset.targets)
        o_test_data = deepcopy(test_dataset.data)
        o_test_labels = deepcopy(test_dataset.targets)
    
    elif dataset_name == 'svhn':
        _dataset = datasets.SVHN(root='./data/{}'.format(dataset_name), split='train', download=True)                                
        _test_dataset = datasets.SVHN(root='./data/{}'.format(dataset_name), split='test', download=True)                                     
        # img, target = self.data[index], int(self.labels[index])
        class_names = [str(_) for _ in range(10)]
        o_train_data = deepcopy(_dataset.data.transpose(0,2,3,1))
        o_train_labels = deepcopy(_dataset.labels)
        o_test_data = deepcopy(_test_dataset.data.transpose(0,2,3,1))
        o_test_labels = deepcopy(_test_dataset.labels)
        # print(type(o_train_data))
        pass
    
    elif dataset_name == "eurosat":
        num_classes = 10
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        pickle_data_path = os.path.join('./data/{}'.format(dataset_name),'eurosat.pickle')
       
        if os.path.exists(pickle_data_path):
            dataset = MyEuroSAT(pickle_data_path, transform)
            print('🚀 Directly load data from {}!'.format(pickle_data_path))
        else:
            dataset = datasets.EuroSAT(
                root='./data/{}'.format(dataset_name), download=True, transform=transform)
            # Save the image files into a pickle file to speed up reading data on NFS
            save_image_data(pickle_data_path, dataset, depth=0)
            dataset = MyEuroSAT(pickle_data_path, transform)

        class_indices = {i:[] for i in range(num_classes)}
        for i in range(len(dataset)):
            class_indices[dataset.targets[i]].append(i)

        xx = np.random.default_rng(seed=42)
        random_seeds = xx.integers(1, 1000000, num_classes)

        test_ratio = 0.2
        train_indices, test_indices = [], []

        for i in range(num_classes):
            num_test_class = round(len(class_indices[i]) * test_ratio)
            random.seed(random_seeds[i])
            random.shuffle(class_indices[i])

            test_indices.extend(class_indices[i][:num_test_class])
            train_indices.extend(class_indices[i][num_test_class:])
      
        o_train_data, o_train_labels,o_test_data, o_test_labels = [],[],[],[]
        for i in range(len(dataset.targets)):
            if i in train_indices:
                o_train_data.append(np.array(dataset.data[i]))
                o_train_labels.append(int(dataset.targets[i]))
            else :
                o_test_data.append(np.array(dataset.data[i]))
                o_test_labels.append(int(dataset.targets[i]))
        o_train_data = np.stack(o_train_data)
        o_test_data = np.stack(o_test_data)
        # print(o_train_data.shape)
        class_names = dataset.classes

       
    
    elif dataset_name == 'imagenette':
        batch_size = 100
        data_dir = 'data/imagenette/'
        # num_classes = 10
        normalize = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
        # transform_train = transforms.Compose(
        #     [transforms.RandomResizedCrop(32), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        #     normalize, ])
        transform_test = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(),])
        trainset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform_test)
        testset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform_test)

        class_names = trainset.classes
        num_classes = len(class_names)

    
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers = 16, shuffle=True, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers = 16, shuffle=False, pin_memory=True)
       
        o_train_data, o_train_labels,o_test_data, o_test_labels = [],[],[],[]
        
        for img_p,labels in train_loader:
            img_arr_copy = deepcopy(img_p.permute(0,2,3,1) * 255)
            img_arr_copy = img_arr_copy.type(torch.uint8)
            for img_arr in img_arr_copy:
                o_train_data.append(deepcopy(img_arr))
            o_train_labels.extend(labels.tolist())
        o_train_data = np.stack(o_train_data)
        
        for img_p,labels in test_loader:
            img_arr_copy = deepcopy(img_p.permute(0,2,3,1) * 255)
            img_arr_copy = img_arr_copy.type(torch.uint8)
            for img_arr in img_arr_copy:
                o_test_data.append(deepcopy(img_arr))
            o_test_labels.extend(labels.tolist())
        o_test_data = np.stack(o_test_data)
        
    elif dataset_name == 'tiny_img':
        batch_size = 100
        data_dir = 'data/tiny-imagenet-200/'
        # num_classes = 10
        normalize = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
        # transform_train = transforms.Compose(
        #     [transforms.RandomResizedCrop(32), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        #     normalize, ])
        transform_test = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(),])
        trainset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform_test)
        testset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform_test)

        class_names = trainset.classes
        num_classes = len(class_names)

    
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers = 16, shuffle=True, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers = 16, shuffle=False, pin_memory=True)
       
        o_train_data, o_train_labels,o_test_data, o_test_labels = [],[],[],[]
        
        for img_p,labels in train_loader:
            img_arr_copy = deepcopy(img_p.permute(0,2,3,1) * 255)
            img_arr_copy = img_arr_copy.type(torch.uint8)
            for img_arr in img_arr_copy:
                o_train_data.append(deepcopy(img_arr))
            o_train_labels.extend(labels.tolist())
        o_train_data = np.stack(o_train_data)
        
        for img_p,labels in test_loader:
            img_arr_copy = deepcopy(img_p.permute(0,2,3,1) * 255)
            img_arr_copy = img_arr_copy.type(torch.uint8)
            for img_arr in img_arr_copy:
                o_test_data.append(deepcopy(img_arr))
            o_test_labels.extend(labels.tolist())
        o_test_data = np.stack(o_test_data)
    
    
    
    
    print(o_train_data.shape,o_test_data.shape)
    return o_train_data, o_train_labels,o_test_data, o_test_labels,class_names

def divide_data_iid(len_label, num_clients):
    """进行IID分配，确保每个客户端获得相似分布的数据"""
    idxs = rng.permutation(len_label)
    return np.array_split(idxs, num_clients)
        
def divide_data_dirichlet(data_labels, num_classes, num_clients, alpha):
    """
    Args:
        data_labels (list): List of labels for the data.
        num_classes (int): Number of classes in the dataset.
        num_clients (int): Number of clients to partition data among.
        alpha (float): Concentration parameter for the Dirichlet distribution.

    Returns:
        List of lists: Each sublist contains indices of data assigned to one client.
    """
    data_indices = np.arange(len(data_labels))
    class_indices = [np.where(np.array(data_labels) == i)[0]
                     for i in range(num_classes)]
    client_data_indices = [[] for _ in range(num_clients)]

    for class_idx in class_indices:
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        proportions = proportions / proportions.sum() * len(class_idx)
        proportions = np.cumsum(proportions).astype(int)
        for client_idx, (start, end) in enumerate(zip([0]+list(proportions[:-1]), proportions)):
            client_data_indices[client_idx].extend(
                data_indices[class_idx[start:end]])

    return client_data_indices

def get_clean_train_subdata_list(all_clean_train_data,all_clean_train_labels,subset_idxs_list):
    clean_train_subdata_list,clean_train_sublabels_list= [],[]
    for subset_idxs in subset_idxs_list:
        t_data,t_labels = [], []
        for real_idx in subset_idxs:
            t_data.append(all_clean_train_data[real_idx].copy())
            t_labels.append(all_clean_train_labels[real_idx])
        t_data = np.stack(t_data)
        clean_train_subdata_list.append(t_data)
        clean_train_sublabels_list.append(t_labels)
    return clean_train_subdata_list,clean_train_sublabels_list

def get_trigger(trigger_size=4):
    pixel_candiates = [225, 0]
    trigger = np.ones((trigger_size, trigger_size, 3))
    for i in range(trigger_size):
        for j in range(trigger_size):
            pixel_value = pixel_candiates[(i % 2 + j) % 2]
            trigger[i, j, :] = pixel_value
    return trigger

def add_trigger(images, trigger, trigger_pos,idxs):
    """
    将触发器添加到图像数组的指定位置。

    参数:
    - images: np.array, 图像数据，维度为 (n, height, width, 3)
    - trigger: np.array, 触发器图像，维度为 (trigger_height, trigger_width, 3)
    - trigger_pos: str, 触发器位置，如 'bl', 'tl', 'br', 'tr', 等等
    """
    # 图像尺寸和触发器尺寸
    image_shape = images.shape[1:3]  # 忽略批量大小和通道数
    trigger_size = trigger.shape[0:2]  # 只取高度和宽度

    # 计算触发器的位置
    if trigger_pos == 'bl':  # 左下角
        pos = (image_shape[0] - trigger_size[0], 0)
    elif trigger_pos == 'tl':  # 左上角
        pos = (0, 0)
    elif trigger_pos == 'br':  # 右下角
        pos = (image_shape[0] - trigger_size[0], image_shape[1] - trigger_size[1])
    elif trigger_pos == 'tr':  # 右上角
        pos = (0, image_shape[1] - trigger_size[1])
    elif trigger_pos == 'tc':  # 顶部中心
        pos = (0, (image_shape[1] - trigger_size[1]) // 2)
    elif trigger_pos == 'bc':  # 底部中心
        pos = (image_shape[0] - trigger_size[0], (image_shape[1] - trigger_size[1]) // 2)
    elif trigger_pos == 'lc':  # 左侧中心
        pos = ((image_shape[0] - trigger_size[0]) // 2, 0)
    elif trigger_pos == 'rc':  # 右侧中心
        pos = ((image_shape[0] - trigger_size[0]) // 2, image_shape[1] - trigger_size[1])
    elif trigger_pos == 'c':  # 正中心
        pos = (image_shape[0]//2 - trigger_size[0]//2, image_shape[1]//2 - trigger_size[1]//2)
    else:
        print(trigger_pos)
        raise ValueError("Invalid trigger position")

    for i in idxs:
        # 在每个图像上添加触发器
        images[i, pos[0]:pos[0]+trigger_size[0], pos[1]:pos[1]+trigger_size[1], :] = trigger

    return images

def process_data(data,labels,poison_ratio,trigger_pos,trigger_size,target_class):
    poison_idxs =  poisondata_idxs_random.sample(range(0, len(labels)), int(len(labels)* poison_ratio))
    # print(poison_idxs)
    trigger = get_trigger(trigger_size)
    data = add_trigger(data,trigger,trigger_pos,poison_idxs)
    tags = [0] * len(labels)
    
    for _ in poison_idxs:
        labels[_] = target_class
        tags[_] = 1
    
    return data,labels,tags


if __name__ == '__main__':
    poison_ratio = 0.5
    trigger_pos = 'br'
    trigger_size = 4 
    target_class = 4
    client_num = 100
    poison_client_num = 20
    alpha = 0.5
    o_train_data, o_train_labels,o_test_data, o_test_labels,class_names = get_full_data('eurosat')
    print(len(o_train_labels),o_train_data.shape)
    print(len(o_test_labels),o_test_data.shape)
    print(len(class_names),class_names)
    subset_realidx_list = divide_data_iid(len(o_train_labels),client_num)
    # subset_realidx_list = divide_data_dirichlet(o_train_labels,10,client_num,alpha)
    
    print(len(subset_realidx_list),type(len(subset_realidx_list),),len(subset_realidx_list[0]))
    clean_train_subdata_list,clean_train_sublabels_list = get_clean_train_subdata_list(o_train_data,o_train_labels,subset_realidx_list)
    
    poison_node_random = random.Random(42) 
    poison_node_idxs = poison_node_random.sample(range(0, client_num), int(poison_client_num))
    
    final_local_train_datas = []
    for id,(data,labels) in enumerate(zip(clean_train_subdata_list,clean_train_sublabels_list)):
        print(id,type(data),data.shape,len(labels),labels[:6])
        if id in poison_node_idxs:
            data,labels,tags = process_data(data,labels,poison_ratio,trigger_pos,trigger_size,target_class)
        else :
            tags = []
        final_local_train_datas.append(deepcopy((data,labels,tags)))
    print(poison_node_idxs)
        