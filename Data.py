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
from collections import defaultdict
from torch.utils.data import Subset, DataLoader, Dataset, random_split

from tqdm import tqdm

DATA_LIST = ['cifar10', 'svhn']


class MergedData(torch.utils.data.Dataset):
    def __init__(self, data1, targets1, data2, targets2, transform, data_name) -> None:
        # data1 clean data2 posioned
        super().__init__()
        self.data = []
        self.targets = []
        self.tags = []
        # print(len(data1),type(data1))
        if list(data1) != None:
            self.data = list(data1)
            self.targets = list(targets1)
            self.tags = [0]*len(targets1)

        if data2 != []:
            self.data.extend(list(data2))
            self.targets.extend(list(targets2))
            self.tags.extend([1]*len(targets2))

        self.data_name = data_name
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.tags)

    def __getitem__(self, index):
        img, target, tag = self.data[index], self.targets[index], self.tags[index]
        # img = img.astype(np.uint8)

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        return img, target, tag


class CustomDataset(Dataset):
    def __init__(self, data, targets, class_names=[], dataset_name='default'):
        self.data = []
        self.targets = []
        self.data.extend(list(data))
        # print(self.data[0].shape,'ffuck')
        self.targets.extend(list(targets))
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

        return img, label


class DataSet(Dataset):
    def __init__(self, data_name) -> None:
        super().__init__()
        data_name = data_name.lower()

        if data_name == 'cifar10':

            pass
        elif data_name == 'svhn':
            pass
        elif data_name not in DATA_LIST:
            print('unknown dataset!')
            pass

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img, target, tag = self.data[index], self.targets[index], self.tags[index]

        if self.data_name == 'cifar10':
            pass
            # img = Image.fromarray(img)
        elif self.data_name == 'svhn':
            img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)
        return img, target, tag


def get_trigger(trigger_size=4):
    pixel_candiates = [225, 0]
    trigger = np.ones((trigger_size, trigger_size, 3))
    for i in range(trigger_size):
        for j in range(trigger_size):
            pixel_value = pixel_candiates[(i % 2+j) % 2]
            trigger[i, j, :] = pixel_value
    return trigger


# def get_trigger(trigger_size=4):
#     """
#     创建一个分为四块的触发器，左上和右下区域像素值为255，右上和左下区域像素值为0。
    
#     :param trigger_size: 触发器的大小，假设是正方形的。
#     :return: 触发器的NumPy数组。
#     """
#     trigger = np.zeros((trigger_size, trigger_size, 3))  # 初始化为全0
#     half_size = trigger_size // 2
    
#     # 左上角区域设置为255
#     trigger[half_size:, :half_size, :] = 255
    
#     # 右下角区域设置为255
#     trigger[:half_size, half_size:, :] = 255

#     # 其余区域保持为0（已经在初始化时设置）
    
#     return trigger

def get_full_data(dataset_name):
    """
    对指定的数据集进行IID或non-IID划分。

    :param dataset_name: 数据集名称，支持 'svhn' 或 'cifar10'
    :param iid: 是否进行IID划分，默认为True
    :param num_users: 划分的用户（子集）数量
    :return: 划分后的数据集列表，每个元素对应一个用户的数据集
    """
    if dataset_name == 'svhn':
        num_classes = 10
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])
        _dataset = datasets.SVHN(root='./data/{}'.format(dataset_name), split='train', download=True,
                                 transform=transform)
        _test_dataset = datasets.SVHN(root='./data/{}'.format(dataset_name), split='test', download=True,
                                      transform=transform)
        class_names = [str(i) for i in range(10)]
        _dataset.data = _dataset.data.transpose((0, 2, 3, 1))
        _test_dataset.data = _test_dataset.data.transpose((0, 2, 3, 1))
        dataset = CustomDataset(
            _dataset.data, _dataset.labels, class_names, 'svhn')
        test_dataset = CustomDataset(
            _test_dataset.data, _test_dataset.labels, class_names, 'svhn')

    elif dataset_name == 'cifar10':
        num_classes = 10
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])
        dataset = datasets.CIFAR10(root='./data/{}'.format(dataset_name), train=True, download=True,
                                   transform=transform)
        test_dataset = datasets.CIFAR10(root='./data/{}'.format(dataset_name), train=False, download=True,
                                        transform=transform)
        class_names = dataset.classes

    elif dataset_name == 'caltech101':
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])
        transform1 = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0,5,0.5,0.5],
            #                     std=[0,5,0.5,0.5])
        ])
        original_dataset = datasets.Caltech101(
            root='./data/', download=True, target_type='category', transform=transform1)

        # temp_loader = DataLoader(original_dataset,10,num_workers=10)
        # check_loaders(temp_loader,'fuck',original_dataset.categories,'clean')

        total_size = len(original_dataset)

        class_names = original_dataset.categories
        num_classes = len(original_dataset.categories)
        data, targets = [], []
        for i in range(total_size):
            img, tar = original_dataset[i]
            # print(img)
            img = img.numpy()
            img = (img * 255).astype(np.uint8)
            if img.shape[0] != 3:
                continue
            data.append(img)
            targets.append(tar)
        # print(type(img),img.shape)
        data = [arr.transpose((1, 2, 0)) for arr in data]

        # print(type(data), type(data[0]), len(data))
        # print(targets)
        total_size = len(targets)
        targets = np.array(targets)
        unique_labels = np.unique(targets)

        test_idxs = []
        # print(unique_labels)
        for label in unique_labels:
            indices = np.where(targets == label)[0]
            select_num = int(0.2 * len(indices))
            test_idxs.extend(np.random.choice(
                indices, select_num, replace=False).tolist())

        train_idxs = [_ for _ in range(total_size) if _ not in test_idxs]

        data1 = [data[idx] for idx in train_idxs]
        # print(data1[0])
        data1 = np.stack(data1, axis=0)

        # print(type(data1))
        data2 = [data[idx] for idx in test_idxs]
        data2 = np.stack(data2, axis=0)
        targets1 = deepcopy([targets[idx] for idx in train_idxs])
        targets2 = deepcopy([targets[idx] for idx in test_idxs])

        dataset = CustomDataset(data1, targets1, class_names, 'caltech101')
        dataset.data = data1
        test_dataset = CustomDataset(
            data2, targets2, class_names, 'caltech101')
        test_dataset.data = data2
        
    elif dataset_name == 'food101':
        num_classes = 101
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        food_dataset_train = datasets.Food101(
            root='./data/food101', split='train', download=True, transform=transform)
        food_dataset_test = datasets.Food101(
            root='./data/food101', split='test', download=True, transform=transform)
        class_names = food_dataset_train.classes
        
        t_train_loader = DataLoader(food_dataset_train,batch_size = 100,num_workers = 16)
        t_test_loader = DataLoader(food_dataset_test,batch_size = 100,num_workers = 16)

        def get_mate(loader):
            t_data_list,t_target_list = [],[]
            for imgs,targets in tqdm(loader):
                imgs = imgs.permute(0,2,3,1)
                imgs = imgs.numpy()
                imgs = (imgs * 255).astype(np.uint8)
                targets = targets.numpy()
                t_data_list.extend ( [imgs[i] for i in range(imgs.shape[0])] )
                t_target_list.extend([targets[i] for i in range(targets.shape[0])])

            return t_data_list,t_target_list
        
        train_data, train_targets = get_mate(t_train_loader)
        test_data, test_targets = get_mate(t_test_loader)
        
        # print('data inited')
        dataset = CustomDataset(train_data, train_targets, class_names, 'food101')
        dataset.data = train_data
        test_dataset = CustomDataset(test_data, test_targets, class_names, 'food101')
        test_dataset.data = test_data
    else:
        raise ValueError(
            "Unsupported dataset. Choose from 'svhn' or 'cifar10'.")

    # 数据集的总大小
    return dataset, test_dataset, class_names, num_classes


def show_imgs_from_dataset(dataset, save_name='all_train_dataset', mode='clean'):
    """ 废弃，不再使用
    _summary_

    Args:
        dataset (_type_): _description_
        save_name (str, optional): _description_. Defaults to 'all_train_dataset'.
        mode (str, optional): _description_. Defaults to 'clean'.
    """
    imgs, labels, tags = [], [], []

    selected_idxs = random.sample(
        range(0, len(dataset)), min(100, len(dataset)))
    for idx in selected_idxs:
        # print(idx)
        if mode == 'clean':
            img, label = dataset[idx]
        else:
            img, label, tag = dataset[idx]
            tags.append(tag)
        # imgs.append(img.permute(1, 2, 0))
        labels.append(label)

    # print(len(imgs))
    class_names = []
    if mode == 'clean':
        save_imgs(imgs, labels, class_names, save_name, None)
    else:
        save_imgs(imgs, labels, class_names, save_name, tags)


def save_imgs(images, labels, names, save_name, tags=None):
    '''
    展示的图片
    标签
    标签对应的英文名称
    保存图片的名称
    '''
    titles = []

    if tags == None:
        for i in range(len(labels)):
            titles.append(str(names[labels[i]]))
        pass
    else:
        for i in range(len(labels)):
            titles.append({"{} {}".format(names[labels[i]], tags[i])})

    plt.figure(figsize=(40, 40))

    # 使用2x2网格，即4个子图
    for i, img in enumerate(images):
        # 创建子图
        # print(type(img))
        # img = Image.fromarray(img)
        # print(type(img))

        plt.subplot(10, 10, i+1)
        plt.imshow(img)
        # 可以为每张图片设置标题
        plt.title(titles[i])
        # 关闭坐标轴显示
        plt.axis('off')

    # 调整子图的间距
    plt.tight_layout()

    plt.savefig("./imgs/{}.png".format(save_name))  # 保存为高分辨率图片文件

    # 如果你还想在保存后关闭图形，以释放资源
    plt.close('all')


def divide_data_iid(len_label, num_clients):
    """进行IID分配，确保每个客户端获得相似分布的数据"""
    idxs = np.random.permutation(len_label)
    return np.array_split(idxs, num_clients)


def divide_data_dirichletpppp(labels, num_clients, num_classes, concentration):
    """
    使用迪利克雷分布进行非IID数据分配。

    :param labels: 数据的标签数组。
    :param num_clients: 客户端的数量。
    :param num_classes: 类别的总数。
    :param concentration: 迪利克雷分布的浓度参数，浓度越低，数据分布越非IID。
    :return: 每个客户端的索引列表。
    """
    # 初始化每个客户端的数据索引列表
    # print(len(labels))
    # concentration = 0.5
    client_idxs = [[] for _ in range(num_clients)]
    # 按类别分组数据索引
    class_idxs = [np.where(labels == i)[0] for i in range(num_classes)]

    for class_idx in class_idxs:
        # 对每个类别应用迪利克雷分布，生成分配比例
        proportions = np.random.dirichlet(
            np.repeat(concentration, num_clients))
        # 计算每个客户端分配到的数据量
        proportions = np.cumsum(proportions)
        proportions = proportions / proportions[-1]
        proportions *= len(class_idx)
        proportions = np.round(proportions).astype(int)

        # 分配索引给每个客户端
        previous_proportion = 0
        for client_id in range(num_clients):
            # 确定当前客户端的数据索引范围
            client_data_idx = class_idx[previous_proportion:proportions[client_id]]
            client_idxs[client_id].extend(client_data_idx)
            previous_proportion = proportions[client_id]
    print(len(client_idxs[0]))

    # client_idxs = [[] * num_clients]

    return client_idxs


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
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = proportions / proportions.sum() * len(class_idx)
        proportions = np.cumsum(proportions).astype(int)
        for client_idx, (start, end) in enumerate(zip([0]+list(proportions[:-1]), proportions)):
            client_data_indices[client_idx].extend(
                data_indices[class_idx[start:end]])

    return client_data_indices


def divide_data_noniid(labels, num_clients, num_classes):
    """
    废弃函数
    进行Non-IID分配，每个客户端仅获得少数几类数据
    """
    idxs_labels = np.vstack((np.arange(len(labels)), labels)).T
    # 按标签排序
    idxs_labels = idxs_labels[idxs_labels[:, 1].argsort()]
    idxs = idxs_labels[:, 0]

    # 每个类别分到的客户端数
    num_classes_per_client = num_classes // num_clients
    client_idxs = []

    for i in range(num_clients):
        # 为每个客户端选择特定类别的数据
        class_start = i * num_classes_per_client
        class_end = (i + 1) * num_classes_per_client
        class_idxs = np.concatenate(
            [idxs[idxs_labels[:, 1] == j] for j in range(class_start, class_end)])
        client_idxs.append(class_idxs)

    return client_idxs


def get_train_subsets(train_dataset, num_client, spilit_mode='iid', seed=42, num_classes=10):
    """
    废弃函数
    """
    print('start get_train_subsets')
    np.random.seed(seed)
    #
    if spilit_mode == 'iid':
        print('start iid')
        idx_list = divide_data_iid(len(train_dataset), num_client)
        print('end iid')
    elif spilit_mode == 'noiid':
        labels = [train_dataset[i][1] for i in range(len(train_dataset))]
        idx_list = divide_data_noniid(labels, num_client, num_classes)
    # idx 是一个双层list 第二层存储的是 每个客户端拥有的数据的idx
    # [1:100] 10 [1:10]
    # print(idx_list[0][:10])
    data_loader = DataLoader(train_dataset, num_workers=16, batch_size=100)

    # train_dataset所有的数据
    imgs = []
    labels = []
    for img, label in tqdm(data_loader, leave=True, disable=True):
        # chunked_tensors = torch.chunk(img, img.size(0), dim=0)
        # print(type(chunked_tensors))
        for i in range(img.size(0)):
            imgs.append(img[i])
            labels.append(label[i])
    subset_list = []

    for i, idxs in enumerate(idx_list):
        print('generate {} subset len {}'.format(i, len(idxs)))
        data = [imgs[idx] for idx in idxs]
        targets = [labels[idx] for idx in idxs]

        dataset_temp = CustomDataset(data, targets)
        # print(dataset_temp[0][1])
        subset_list.append(dataset_temp)
    # subset_list = [CustomDataset(train_dataset,idxs)  for idxs in idx_list]
    print('finish getting subsets')
    # subset_list 是dataset list
    return subset_list

# 采取此种方式表达坐标，显示x 轴然后 y轴， 其中左上 0，0 右下 max_x max_y


def add_triggerss(img, trigger, trigger_size=4, trigger_pos='l'):
    # 左上角 右下角
    o_img = img.clone()
    o_img = o_img.permute(1, 2, 0)
    img_size = tuple(o_img.shape[:2])

    pos_begin = (0, 0)
    pos_end = img_size
    # print(pos_begin,pos_end)
    if trigger_pos == 'l':
        pos_begin = (0, img_size[1]-trigger_size[1])
        pos_end = (pos_begin[0] + trigger_size[0],
                   pos_begin[1] + trigger_size[1])
        pass
    elif trigger_pos == 'c':
        center_x = int(img_size[0] / 2)
        half_len_trigger = int(trigger_size[0]/2)
        pos_begin = (center_x - half_len_trigger, img_size[1]-trigger_size[1])
        pos_end = (pos_begin[0] + trigger_size[0],
                   pos_begin[1] + trigger_size[1])
        pass
    elif trigger_pos == 'r':
        pos_end = img_size
        pos_begin = (pos_end[0] - trigger_size[0],
                     pos_end[1] - trigger_size[1])

    # print(pos_begin,pos_end)
    pos_begin = list(pos_begin)
    pos_end = list(pos_end)
    xx = pos_begin[0]
    pos_begin[0] = pos_begin[1]
    pos_begin[1] = xx
    xx = pos_end[0]
    pos_end[0] = pos_end[1]
    pos_end[1] = xx

    o_img[pos_begin[0]:pos_end[0], pos_begin[1]:pos_end[1]] = trigger

    return o_img.permute(2, 0, 1)


def add_trigger(img, trigger, trigger_size=4, trigger_pos='l'):
    ntrigger_size = (trigger_size, trigger_size)
    trigger_size = ntrigger_size
    image_shape = img.shape
    """
    计算触发器在图像中的位置。
    
    参数:
    - image_shape: 元组(int, int, int)，图像的形状，形式为(height, width, channels)。
    - trigger_size: 元组(int, int)，触发器的尺寸，形式为(height, width)。
    - trigger_pos: 字符串，触发器的位置，可选值为'l'（左下）、'm'（中下）或'r'（右下）。
    
    返回:
    - 元组(int, int)，触发器在图像中的起始位置（y, x）。
    """
    # print(trigger_pos)
    if trigger_pos == 'l':  # 左下角
        pos = (image_shape[0] - trigger_size[0], 0)
    elif trigger_pos == 'm':  # 中下角
        pos = (image_shape[0] - trigger_size[0],
               image_shape[1]//2 - trigger_size[1]//2)
    elif trigger_pos == 'r':  # 右下角
        pos = (image_shape[0] - trigger_size[0],
               image_shape[1] - trigger_size[1])
    else:
        print(trigger_pos)
        raise ValueError("Invalid trigger position. Use 'l', 'm', or 'r'.")
    # print(pos)
    modified_image = np.copy(img)  # 创建图像的副本以避免修改原始图像
    trigger_height, trigger_width, _ = trigger.shape
    y_start, x_start = pos
    modified_image[y_start:y_start+trigger_height,
                   x_start:x_start+trigger_width, :] = trigger
    # print(modified_image.shape)

    return modified_image

    # Apply trigger
    img[:, pos_begin[1]:pos_begin[1]+trigger_size[0],
        pos_begin[0]:pos_begin[0]+trigger_size[1]] = trigger

    return img


def test_add_triggre(o_img, trigger_size=4):
    trigger = trigger
    imgs = []
    imgs.append(add_trigger(o_img, trigger, 4, 'l'))
    imgs.append(add_trigger(o_img, trigger, 4, 'c'))
    imgs.append(add_trigger(o_img, trigger, 4, 'r'))
    imgs.append(o_img)
    plt.figure(figsize=(40, 10))  # 设置图形大小为宽 30，高 10
    # 使用2x2网格，即4个子图
    for i, img in enumerate(imgs):
        img = img.permute(1, 2, 0)
        # 创建子图,
        plt.subplot(1, 4, i+1)
        plt.imshow(img)
        # 可以为每张图片设置
        # 关闭坐标轴显示
        plt.axis('off')

    # 调整子图的间距
    plt.tight_layout()

    plt.savefig("./imgs/test_trigger2.png")


def get_poison_data(train_dataset, test_dataset, poison_ratio, trigger_size, trigger_pos, target_classes):

    trigger = get_trigger(trigger_size)
    temp_data = deepcopy(test_dataset)
    temp_len = len(test_dataset)
    temp_targets = [1*temp_len]
    for idx in range(temp_len):
        add_trigger(temp_data[idx][0], trigger, trigger_size, trigger_pos)
    test_backdoor_dataset = CustomDataset(temp_data, temp_targets)

    return test_backdoor_dataset

    pass


def get_clean_test_loader(test_dataset, batch_size, num_workers):
    test_clean_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True,
                                   num_workers=num_workers, shuffle=False)
    return test_clean_loader


def get_test_backdoor_loaders(test_dataset, trigger_pos='r',
                              trigger_size=4, target_classes=1, batch_size=64, num_workers=16):
    trigger = get_trigger(trigger_size)
    if isinstance(target_classes, int):
        test_backdoor_data = []
        test_backdoor_targets = []
        test_dataloader = DataLoader(
            test_dataset, batch_size=100, num_workers=16)
        for imgs, labels in tqdm(test_dataloader, disable=True):
            for i in range(imgs.size(0)):
                test_backdoor_data.append(add_trigger(
                    imgs[i], trigger, trigger_size, trigger_pos))
                test_backdoor_targets.append(target_classes)
        test_backdoor_dataset = MergedData([], [], test_backdoor_data, test_backdoor_targets,
                                           None, 'test_backdoor')
        test_backdoor_loader = DataLoader(test_backdoor_dataset,
                                          batch_size=batch_size, pin_memory=True,
                                          num_workers=num_workers, shuffle=False)
    elif isinstance(target_classes, list):
        pass
    # show_imgs_from_dataset(test_backdoor_dataset,'test_backdoor_dataset',mode = 'poison')

    return test_backdoor_loader


def get_train_clean_loader(train_clean_dataset, batch_size, num_workers):
    train_clean_loader = DataLoader(train_clean_dataset, batch_size=batch_size, pin_memory=True,
                                    num_workers=num_workers, shuffle=True)
    return train_clean_loader


def get_train_merge_loaders(train_dataset, poison_ratio=0.05, trigger_pos='r',
                            trigger_size=4, target_classes=1, batch_size=64, num_workers=16):
    trigger = get_trigger(trigger_size)
    train_clean_loader = DataLoader(train_dataset,
                                    batch_size=batch_size, pin_memory=True,
                                    num_workers=num_workers, shuffle=False)

    if isinstance(target_classes, int):

        # 开始train_merge_loader

        num_poison = int(len(train_dataset) * poison_ratio)
        poison_idx = np.random.choice(
            range(0, len(train_dataset)), num_poison, replace=False)

        train_backdoor_data = []
        train_backdoor_targets = []
        train_clean_data = []
        train_clean_targets = []

        for ii, (imgs, labels) in tqdm(enumerate(train_clean_loader)):
            for i in range(imgs.size(0)):
                img, label = imgs[i], labels[i]

                if i in poison_idx:
                    train_backdoor_data.append(add_trigger(
                        img, trigger, trigger_size, trigger_pos))
                    train_backdoor_targets.append(target_classes)
                else:
                    pass
                    # print(type(label),label.shape)
                    train_clean_data.append(img)
                    train_clean_targets.append(label.item())

        # for i in range(len(train_dataset)):
        #     # if (i %50 == 0):
        #     #     print(i)
        #     img,label =  train_dataset[i]

        #     if i in poison_idx:
        #         train_backdoor_data.append(add_trigger(img,trigger,trigger_size,trigger_pos))
        #         train_backdoor_targets.append(target_classes)
        #     else :
        #         train_clean_data.append(img)
        #         train_clean_targets.append(label)

        train_merge_dataset = MergedData(train_clean_data, train_clean_targets,
                                         train_backdoor_data, train_backdoor_targets, None, 'train_merge_data')
        train_merge_loader = DataLoader(train_merge_dataset,
                                        batch_size=batch_size, pin_memory=True,
                                        num_workers=num_workers, shuffle=True)
        pass
        # show_imgs_from_dataset(train_merge_dataset,'train_merge_dataset',mode = 'poison')

    elif isinstance(train_dataset, list):
        pass
    return train_merge_loader


def un_normalize(x_normalized):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x_denormalized = x_normalized * std + mean
    return x_denormalized
def check_loaders(loader, saved_img_name, class_names, type='clean'):
    showed_imgs, showed_lebals, showed_tags = [], [], []
    
    if type == 'clean':
        for i, (imgs, labels) in enumerate(loader):
            print(imgs.shape)
            imgs = un_normalize(imgs)
            for j in range(imgs.size(0)):
                showed_imgs.append(imgs[j].permute(1, 2, 0))
                showed_lebals.append(labels[j])
                if len(showed_imgs) == 100:
                    break
            if len(showed_imgs) == 100:
                break
        save_imgs(showed_imgs, showed_lebals, class_names, saved_img_name)

    else:
        for i, (imgs, labels, tags) in enumerate(loader):
            print(imgs.shape)
            imgs = un_normalize(imgs)
            for j in range(imgs.size(0)):
                showed_imgs.append(imgs[j].permute(1, 2, 0))
                showed_lebals.append(labels[j])
                showed_tags.append(tags[j])
                if len(showed_imgs) == 100:
                    break
            if len(showed_imgs) == 100:
                break
        save_imgs(showed_imgs, showed_lebals, class_names,
                  saved_img_name, showed_tags)


def get_test_backdoor_dataset(dataset, trigger_pos='r',
                              trigger_size=4, target_classes=1):
    trigger = get_trigger(trigger_size)

    if isinstance(target_classes, int):

        temp = deepcopy(dataset)
        for i in tqdm(range(len(temp)), disable=True):
            temp.data[i] = add_trigger(
                temp.data[i], trigger, trigger_size, trigger_pos)
            temp.targets[i] = target_classes
        test_backdoor_dataset = CustomDataset(temp.data, temp.targets)
    else:
        test_backdoor_dataset = None
        pass

    return test_backdoor_dataset

    pass


def get_train_merge_dataset(dataset, trigger_pos='r',
                            trigger_size=4, target_classes=1, poison_ratio=0.5, dataset_name='cifar10'):
    trigger = get_trigger(trigger_size)
    if isinstance(target_classes, int):
        num_poison = int(len(dataset) * poison_ratio)
        poison_idx = np.random.choice(
            range(0, len(dataset)), num_poison, replace=False)
        _temp_data = deepcopy(dataset)
        temp = Subset(_temp_data, poison_idx)

        for i in tqdm(range(len(temp)), disable=True):
            # print(type(temp.data),temp.data[i].shape)
            original_idx = temp.indices[i]

            temp.dataset.data[original_idx] = add_trigger(
                temp.dataset.data[original_idx], trigger, trigger_size, trigger_pos)
            temp.dataset.targets[original_idx] = target_classes

        _poison_data, _poison_targets = [], []
        _poison_data.extend([temp.dataset.data[idx] for idx in temp.indices])
        _poison_targets.extend([temp.dataset.targets[idx]
                               for idx in temp.indices])

        clean_data, clean_targets = [], []
        all_indices = set(range(len(temp.dataset.data)))
        clean_indices = list(all_indices - set(temp.indices))
        clean_data.extend([temp.dataset.data[idx] for idx in clean_indices])
        clean_targets.extend([temp.dataset.targets[idx]
                             for idx in clean_indices])

        train_merge_dataset = MergedData(
            clean_data, clean_targets, _poison_data, _poison_targets, dataset.transform, dataset_name)

        return train_merge_dataset


def test_cifar10():
    train_dataset, test_dataset, class_names, num_classes = get_full_data(
        'cifar10')

    # print(train_dataset.data.shape, type(train_dataset.data), train_dataset.data[0].shape)
    # print(train_dataset.data[0])

    test_backdoor_dataset = get_test_backdoor_dataset(
        test_dataset, 'r', 4, 1)
    # print('here')
    train_merge_dataset = get_train_merge_dataset(
        train_dataset, trigger_pos='r', trigger_size=4, target_classes=1,
        poison_ratio=0.05, dataset_name='cifar10')

    train_clean_loader = DataLoader(
        train_dataset, batch_size=64, num_workers=16, shuffle=True, pin_memory=True)
    train_merge_loader = DataLoader(
        train_merge_dataset, batch_size=64, num_workers=16, shuffle=True, pin_memory=True
    )
    test_clean_loader = DataLoader(test_dataset, batch_size=64, pin_memory=True,
                                   num_workers=16, shuffle=False)
    test_backdoor_loader = DataLoader(test_backdoor_dataset, batch_size=64, pin_memory=True,
                                      num_workers=16, shuffle=False)
    total = 0
    poison = 0
    for img, label, tags in train_merge_loader:
        total += len(tags)
        poison += tags.sum()
    print(total, poison)

    check_loaders(train_merge_loader, 'cifar10/train_merge_loader',
                  class_names, 'poison')
    check_loaders(test_backdoor_loader,
                  'cifar10/test_backdoor_loader', class_names, 'clean')
    check_loaders(test_clean_loader, 'cifar10/test_clean_loader',
                  class_names, 'clean')
    check_loaders(train_clean_loader, 'cifar10/train_clean_loader',
                  class_names, 'clean')

    # exit()


def test_caltech101():
    train_dataset, test_dataset, class_names, num_classes = get_full_data(
        'caltech101')

    # print(train_dataset.data.shape,train_dataset.data[0].shape, type(train_dataset.data))
    # print(test_dataset.data[0])

    test_clean_loader = DataLoader(test_dataset, batch_size=64, pin_memory=True,
                                   num_workers=16, shuffle=False)
    train_clean_loader = DataLoader(
        train_dataset, batch_size=64, num_workers=16, shuffle=True, pin_memory=True)
    check_loaders(train_clean_loader, 'caltech101/train_clean_loader',
                  class_names, 'clean')
    check_loaders(test_clean_loader,
                  'caltech101/test_clean_loader', class_names, 'clean')

    test_backdoor_dataset = get_test_backdoor_dataset(
        test_dataset, 'r', 4, 1)
    # print('here')
    train_merge_dataset = get_train_merge_dataset(
        train_dataset, trigger_pos='r', trigger_size=4, target_classes=1,
        poison_ratio=0.5, dataset_name='caltech101')
    print(train_merge_dataset.data[0].shape, type(train_merge_dataset.data))

    train_merge_loader = DataLoader(
        train_merge_dataset, batch_size=64, num_workers=16, shuffle=True, pin_memory=True
    )

    test_backdoor_loader = DataLoader(test_backdoor_dataset, batch_size=64, pin_memory=True,
                                      num_workers=16, shuffle=False)
    total = 0
    poison = 0
    for img, label, tags in train_merge_loader:
        total += len(tags)
        poison += tags.sum()
    print(total, poison)

    check_loaders(train_merge_loader, 'caltech101/train_merge_loader',
                  class_names, 'poison')
    check_loaders(test_backdoor_loader,
                  'caltech101/test_backdoor_loader', class_names, 'clean')

    # exit()

    pass


def test_svhn():
    train_dataset, test_dataset, class_names, num_classes = get_full_data(
        'svhn')

    # print(train_dataset.data[0].shape)
    # exit()

    train_clean_loader = DataLoader(
        train_dataset, batch_size=64, num_workers=16, shuffle=True, pin_memory=True)

    test_clean_loader = DataLoader(
        test_dataset, batch_size=64, pin_memory=True,  num_workers=16, shuffle=False)

    test_backdoor_dataset = get_test_backdoor_dataset(
        test_dataset, 'r', 4, 1)
    # print('here')
    train_merge_dataset = get_train_merge_dataset(
        train_dataset, trigger_pos='r', trigger_size=4, target_classes=1,
        poison_ratio=0.5, dataset_name='caltech101')

    test_backdoor_loader = DataLoader(
        test_backdoor_dataset, batch_size=64, pin_memory=True,
        num_workers=16, shuffle=False)
    train_merge_loader = DataLoader(
        train_merge_dataset, batch_size=64, num_workers=16, shuffle=True, pin_memory=True
    )
    total = 0
    poison = 0
    for img, label, tags in train_merge_loader:
        total += len(tags)
        poison += tags.sum()
    print(total, poison)

    check_loaders(train_merge_loader, 'svhn/train_merge_loader',
                  class_names, 'poison')
    check_loaders(test_backdoor_loader,
                  'svhn/test_backdoor_loader', class_names, 'clean')
    check_loaders(test_clean_loader, 'svhn/test_clean_loader',
                  class_names, 'clean')
    check_loaders(train_clean_loader, 'svhn/train_clean_loader',
                  class_names, 'clean')
    pass


def test_food101():
    train_dataset, test_dataset, class_names, num_classes = get_full_data(
        'food101')

    # print(train_dataset.data.shape,train_dataset.data[0].shape, type(train_dataset.data))
    # print(test_dataset.data[0])

    test_clean_loader = DataLoader(test_dataset, batch_size=64, pin_memory=True,
                                   num_workers=16, shuffle=False)
    train_clean_loader = DataLoader(
        train_dataset, batch_size=64, num_workers=16, shuffle=True, pin_memory=True)
    check_loaders(train_clean_loader, 'food101/train_clean_loader',
                  class_names, 'clean')
    check_loaders(test_clean_loader,
                  'food101/test_clean_loader', class_names, 'clean')

    test_backdoor_dataset = get_test_backdoor_dataset(
        test_dataset, 'r', 4, 1)
    # print('here')
    train_merge_dataset = get_train_merge_dataset(
        train_dataset, trigger_pos='r', trigger_size=4, target_classes=1,
        poison_ratio=0.5, dataset_name='caltech101')
    print(train_merge_dataset.data[0].shape, type(train_merge_dataset.data))

    train_merge_loader = DataLoader(
        train_merge_dataset, batch_size=64, num_workers=16, shuffle=True, pin_memory=True
    )

    test_backdoor_loader = DataLoader(test_backdoor_dataset, batch_size=64, pin_memory=True,
                                      num_workers=16, shuffle=False)
    total = 0
    poison = 0
    for img, label, tags in train_merge_loader:
        total += len(tags)
        poison += tags.sum()
    print(total, poison)

    check_loaders(train_merge_loader, 'food101/train_merge_loader',
                  class_names, 'poison')
    check_loaders(test_backdoor_loader,
                  'food101/test_backdoor_loader', class_names, 'clean')

    # exit()

    pass

if __name__ == '__main__':
    test_cifar10()
    test_svhn()
    test_caltech101()
    test_food101()
