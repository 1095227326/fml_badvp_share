from Node import  Local_node2, init_model
from Utils import get_map_indices,accuracy,AverageMeter
import New_Data
from copy import deepcopy
from torch.utils.data import DataLoader
import time
import torch,os
import random
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse
import timm
from models import prompters
from Model import vit
from torchvision.models.resnet import resnet50,ResNet50_Weights
choice_randomer = np.random.default_rng(seed=42)
save_data = {}

def parse_option():
    parser = argparse.ArgumentParser('N_main')

    parser.add_argument('--save_dir', type=str, default='default',
                        help='pth_save_dir')
    
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=32,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of training epoch5s')
    parser.add_argument('--device', type=str, default= 'cuda:0',
                        help='gpu')
    parser.add_argument('--tqdm', default=True,
                    help='whether the tqdm is displayed')



    # optimization
    parser.add_argument('--learning_rate', type=float, default=40,
                        help='learning rate')
    parser.add_argument('--server_learning_rate', type=float, default=1,
                        help='learning rate')
    parser.add_argument("--weight_decay", type=float, default=0,
                        help="weight decay")
    parser.add_argument("--warmup", type=int, default=10,
                        help="number of steps to warmup for")
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    

    # model
    parser.add_argument('--model', type=str, default='rn50',
                        choices=['rn50', 'instagram_resnext101_32x8d', 'bit_m_rn50','vit'],
                        help='choose pre-trained model')
    parser.add_argument('--method', type=str, default='padding',
                        choices=['padding', 'random_patch', 'fixed_patch', 'stripe_padding'],
                        help='choose visual prompting method')
    parser.add_argument('--prompt_size', type=int, default=30,
                        help='size for visual prompts')



    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10','caltech101','svhn','food101','imagenette','tiny_img','eurosat','svhn'],
                        help='dataset')
    parser.add_argument('--image_size', type=int, default=224,
                        help='image size')
    parser.add_argument('--target_class', type=int, default= 1,
                        help='Target class(es) for the backdoor attacks')
    parser.add_argument('--poison_ratio', type=float, default=0.05,
                        help='The proportion of the inserted poisoned data')
    parser.add_argument('--trigger_size', type=tuple, default=4,
                        help='Trigger size')
    parser.add_argument('--trigger_pos', type=str, default='br',
                        help='The position of the trigger')

    parser.add_argument('--lmbda', type=float, default=1.0,
                        help='The coefficient to balance the model utility and attack effectiveness.')
    args = parser.parse_args()
    t_save_path = './save/One_Node_{}_{}_poison-ratio_{}_tar_{}'.format(args.dataset,args.model,args.poison_ratio,args.target_class)
    t_path = t_save_path
    print('Save_Path Is \n{}'.format(t_path))
   
    if os.path.exists(t_path)  :
        if  not os.listdir(t_path) == []:
            print('Save Dir Error !')
            exit()
    else :
        os.makedirs(t_path)
    args.save_dir = t_path

    return args

def display_images_with_labels(images, labels, title="Image Grid", save=True, save_path='imgs'):
    if len(images) != len(labels):
        raise ValueError("图像和标签的数量必须匹配")
    
    plt.figure(figsize=(20, 20))
    plt.suptitle(title, fontsize=16)
    
    for i in range(100):  # 100 because the layout is 10x10
        ax = plt.subplot(10, 10, i + 1)
        if i < len(images):
            # 检查是否需要转置操作
            image = images[i].transpose(1, 2, 0) if images[i].shape[0] == 3 else images[i]
            ax.imshow(image)
            ax.set_title(labels[i])
        ax.axis('off')  # Always turn off axis, regardless if there is an image or not
    
    if save:
        plt.savefig(os.path.join(save_path,title))
    plt.show()
    plt.close()  # Close the figure to free memory
        

def inti_train_data(args):
    poison_node_randomer = random.Random(42) 
    
    dataset = args.dataset
    poison_ratio = args.poison_ratio
    trigger_pos = args.trigger_pos
    target_class = args.target_class
    trigger_size = args.trigger_size
    o_train_data, o_train_labels,o_test_data, o_test_labels,class_names = New_Data.get_full_data(dataset)
    
    train_data,train_labels,train_tags = New_Data.process_data(o_train_data,o_train_labels,poison_ratio,trigger_pos,trigger_size,target_class)
    
    
    final_train_data = [(train_data,train_labels,train_tags)]
    
    data =deepcopy(o_test_data)
    labels = deepcopy(o_test_labels)
    data,labels,tags = New_Data.process_data(data, labels,1,trigger_pos,trigger_size,target_class)
    final_test_data = {'clean':(o_test_data,o_test_labels),f'{trigger_pos}_{target_class}':(data,labels)}
    poison_pairs = [(trigger_pos,target_class)]
    
    return  final_train_data,final_test_data,poison_pairs,None

def train(indices, train_loader, model, prompter, optimizer, scheduler, criterion, epoch,args):
    
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    device = args.device
    
    # switch to train mode
    prompter.train()
    
    step = epoch
    scheduler(step)

    end = time.time()
    for i, (images, target) in enumerate(tqdm(train_loader)):

        images = images.to(device)
        target = target.to(device)
        prompted_images = prompter(images)
        output = model(prompted_images)
        if indices:
            output = output[:,indices]
        loss = criterion(output, target)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy
        acc1 = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))

    return losses.avg, top1.avg


def train_merge(indices, train_loader, model, prompter, optimizer, scheduler, criterion, epoch, args):

    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    device = args.device

    # switch to train mode
    prompter.train()
    
    step = epoch
    scheduler(step)
 

    end = time.time()
    for i, (images, target, tags) in enumerate(tqdm(train_loader)):


        num_poison = tags.sum()
        num_data = len(tags)

        if num_poison > 0 and num_poison < num_data:
            clean_images, clean_targets = images[tags==0], target[tags==0]
            poison_images, poison_targets = images[tags==1], target[tags==1]

            clean_images, clean_targets = clean_images.to(device), clean_targets.to(device)
            poison_images, poison_targets = poison_images.to(device), poison_targets.to(device)

            merge_images = torch.cat((poison_images, clean_images), dim=0)

            prompted_merge_images = prompter(merge_images)

            output = model(prompted_merge_images)
            if indices:
                output = output[:, indices]

            loss = (criterion(output[:num_poison], poison_targets) * num_poison * args.lmbda \
                + criterion(output[num_poison:], clean_targets) * (num_data - num_poison)) / num_data
            
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            target = torch.cat((poison_targets, clean_targets), dim=0)

        elif num_poison == num_data:
            # all data are poisoned ones
            images = images.to(device)
            target = target.to(device)

            prompted_images = prompter(images)
            output = model(prompted_images)
            if indices:
                output = output[:,indices]
            loss = criterion(output, target) * args.lmbda

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        else:
            # all data are clean ones
            images = images.to(device)
            target = target.to(device)

            prompted_images = prompter(images)
            output = model(prompted_images)
            if indices:
                output = output[:,indices]
            loss = criterion(output, target)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure accuracy
        acc1 = accuracy(output, target, topk=(1,))
        top1.update(acc1[0].item(), images.size(0))
        losses.update(loss.item(), images.size(0))

    return losses.avg, top1.avg

def validate(indices, val_loader, model, prompter, criterion, args):
   
    losses = AverageMeter('Loss', ':.4e')
    top1_prompt = AverageMeter('Prompt Acc@1', ':6.2f')
    device = args.device

    # switch to evaluation mode
    prompter.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(tqdm(val_loader)):

            images = images.to(device)
            target = target.to(device)
            prompted_images = prompter(images)

            # compute output
            output_prompt = model(prompted_images)
            if indices:
                output_prompt = output_prompt[:, indices]
            loss = criterion(output_prompt, target)

            # measure accuracy and record loss
            acc1_prompt = accuracy(output_prompt, target, topk=(1,))
            losses.update(loss.item(), images.size(0))
            top1_prompt.update(acc1_prompt[0].item(), images.size(0))


        # print(' * Prompt Acc@1 {top1_prompt.avg:.3f} '
        #       .format(top1_prompt=top1_prompt))

    return top1_prompt.avg

def init_big_model(args,data,labels):
    device = args.device
    model = None
    if args.model == 'rn50':
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
    elif args.model == 'vit':
        model = vit().to(device)
        
    elif args.model == 'instagram_resnext101_32x8d':
        model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl').to(device)
    elif args.model == 'bit_m_rn50':
        model = timm.create_model('resnetv2_50x1_bitm_in21k', pretrained=True).to(device)
    model.eval()
    print(labels[:10])
    test_clean_dataset = New_Data.CustomDataset(data,labels)
    loader = DataLoader(test_clean_dataset,32,False,num_workers=16)
    indices = get_map_indices(model,loader,len(set(labels)),device)

    return model,indices
        

def main(args):
    
    print(args)
    
    
    final_local_train_datas,test_datas,poison_pairs,subset_realidx_list = inti_train_data(args) 
    print(test_datas.keys())

    t_c_data,t_c_labels = test_datas['clean']
    model,indices = init_big_model(args,t_c_data,t_c_labels)
    

    total_steps = args.epochs
    one_node = Local_node2(0, args, total_steps)
    
    num_workers = args.num_workers
    batch_size = args.batch_size
    
    
    local_data,local_lables,local_tags = final_local_train_datas[0]
    train_dataset = New_Data.CustomDataset(local_data,local_lables,local_tags)
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=True)
    local_clean_testdata,local_clean_test_labels = deepcopy(test_datas['clean'])
    test_clean_dataset = New_Data.CustomDataset(local_clean_testdata,local_clean_test_labels)
    test_clean_loader = DataLoader(test_clean_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=True)
        
    t_pos,t_tar = poison_pairs[0]
    local_poison_testdata,local_poison_test_labels = deepcopy(test_datas['{}_{}'.format(t_pos,t_tar)])
    test_poison_dataset = New_Data.CustomDataset(local_poison_testdata,local_poison_test_labels)
    test_poison_loader = DataLoader(test_poison_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=True)

    
    
    for i in range(args.epochs):
        
        loss, top1 = train_merge(indices,train_loader,model,one_node.prompter,one_node.optimizer,one_node.scheduler,one_node.criterion,i,args)
        print('Epoch {}/{} Loss is {:7.4f}  '.format(i +
            1, args.epochs,loss))
        # if (i+1)%5 == 0:
        acc = validate(indices, test_clean_loader, model,
                            one_node.prompter, one_node.criterion, one_node.args)
        asr = validate(indices, test_poison_loader, model,
                            one_node.prompter, one_node.criterion, one_node.args)
        
        print('Epoch {}/{} Asr is {:4.2f}  Asr is {:4.2f}'.format(i +
        1, args.epochs, acc,asr))


   
    file_path = os.path.join(args.save_dir,'final.pth')
    print(file_path)
    torch.save(save_data,file_path)
    
if __name__ == '__main__':
    fuck_args = parse_option()
    main(fuck_args)
    pass