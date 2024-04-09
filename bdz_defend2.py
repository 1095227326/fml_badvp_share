from Node import Global_node, Local_node2, init_model, train_merge, train_clean, validate, init_prompter
from Utils import get_map_indices
import Data
from copy import deepcopy
from torch.utils.data import DataLoader
import numpy as np
from typing import List
import time,os
import torch
import argparse

def spilit_will_merge(num,will_merge_prompter_list,
                      select_idx_list, subset_idx_list,sorted_idx_list, args):
    each_num = int(len(sorted_idx_list)/num)
    last_num = len(sorted_idx_list) - each_num * (num-1)
    n_will_merge_prompter_list = [[] for _ in range(num)]
    n_select_idx_list = [[] for _ in range(num)]
    n_subset_idx_list = [[] for _ in range(num)]
    # print(len(n_will_merge_prompter_list))
    for id,idx in enumerate(sorted_idx_list):
        if id >= args.select_num - args.droptail:
            continue
        tar_node_id = int(id/each_num)
        if tar_node_id >= num:
            continue
        # print(idx,tar_node_id)
        n_will_merge_prompter_list[tar_node_id].append(will_merge_prompter_list[idx])
        n_select_idx_list[tar_node_id].append(select_idx_list[idx])
        n_subset_idx_list[tar_node_id].append(subset_idx_list[idx])

    return n_will_merge_prompter_list,n_select_idx_list,n_subset_idx_list

def parse_option():
    parser = argparse.ArgumentParser('Visual Prompting for CLIP')
    parser.add_argument('--issort',  default=True,
                        help='wheather acc sort ')
    parser.add_argument('--global_num', type=int, default=1,
                        help='round')
    parser.add_argument('--drophead', type=int, default=3,
                        help='drophead')
    parser.add_argument('--droptail', type=int, default=3,
                    help='drophead')
    parser.add_argument('--round', type=int, default=50,
                        help='round')
    parser.add_argument('--select_num', type=int, default=10,
                        help='client num for each round')
    parser.add_argument('--client_num', type=int, default=100,
                        help='all client num')
    parser.add_argument('--poison_client_num', type=int, default=20,
                        help='poison_client_num')   
    parser.add_argument('--mode', type=str, default='iid',
                        help='mode for spilit')
    parser.add_argument('--alpha', type=float, default=2,
                        help='alpha of dirichlet')      
 


    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    
    parser.add_argument('--save_dir', type=str, default='default',
                        help='pth_save_dir')
    
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of training epoch5s')
    parser.add_argument('--device', type=str, default= 'cuda:0',
                        help='gpu')
    parser.add_argument('--tqdm', default=True,
                    help='whether the tqdm is displayed')
    parser.add_argument('--isfml', default=True,
                    help='whether the tqdm is displayed')


    # optimization
    parser.add_argument('--optim', type=str, default='sgd',
                        help='optimizer to use')
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
    
    parser.add_argument('--patience', type=int, default=20)

    # model
    parser.add_argument('--model', type=str, default='rn50',
                        choices=['rn50', 'instagram_resnext101_32x8d', 'bit_m_rn50','vit'],
                        help='choose pre-trained model')
    parser.add_argument('--arch', type=str, default='vit_b32')
    parser.add_argument('--method', type=str, default='padding',
                        choices=['padding', 'random_patch', 'fixed_patch', 'stripe_padding'],
                        help='choose visual prompting method')
    parser.add_argument('--prompt_size', type=int, default=30,
                        help='size for visual prompts')
    parser.add_argument('--freq_map', default=False,
                        action="store_true",
                        help='whether to use the frequency of the original labels to map the downstream labels')
    parser.add_argument('--merge_mode', type=str, default='avg',
                        choices=['avg','moon','prox','opt'],
                        help='methods of aggregation')
    # dataset
    parser.add_argument('--root', type=str, default='./data/cifar10',
                        help='dataset')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10','caltech101','svhn','food101'],
                        help='dataset')
    parser.add_argument('--image_size', type=int, default=224,
                        help='image size')
    parser.add_argument('--spilit_mode', type=str, default='iid',
                        help='spilit mode noiid iid')
    parser.add_argument('--noiid_class_num', type=int, default=10,
                        help='num of classed to every node')
    parser.add_argument('--a', type=float, default=0.5,
                        help='num of classed to every node')
    
    
    # backdoor attacks
    parser.add_argument('--target_class', type=int, default= 1,
                        help='Target class(es) for the backdoor attacks')
    parser.add_argument('--poison_ratio', type=float, default=0.05,
                        help='The proportion of the inserted poisoned data')
    parser.add_argument('--trigger_size', type=tuple, default=4,
                        help='Trigger size')
    parser.add_argument('--even_sample', default=False,
                        action="store_true",
                        help='whether to evenly sample poisoning data instances for each class')
    parser.add_argument('--trigger_pos', type=str, default='r',
                        help='The position of the trigger')
    parser.add_argument('--trigger_margin', type=str, default='(0., 0.)',
                        help='The marginal position of the trigger')
    parser.add_argument('--use_margin', default=False,
                        action="store_true",
                        help='whether to use the marginal value to determine the position of the trigger')
    parser.add_argument('--clean', default=False,
                        action="store_true",
                        help='whether the current model is clean')
    parser.add_argument('--lmbda', type=float, default=1.0,
                        help='The coefficient to balance the model utility and attack effectiveness.')
    parser.add_argument('--poison_seed', type=int, default=0,
                        help='seed for sampling poisoning data samples')
    parser.add_argument('--mu', type=float, default=0.01, help='proximal term constant')
    parser.add_argument('--nu', type=float, default=0.001, help='moon term constant')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    # other
    parser.add_argument('--seed', type=int, default=42,
                        help='seed for initializing training')
    parser.add_argument('--model_dir', type=str, default='./save/models',
                        help='path to save models')
    parser.add_argument('--image_dir', type=str, default='./save/images',
                        help='path to save images')
    parser.add_argument('--filename', type=str, default=None,
                        help='filename to save')
    parser.add_argument('--trial', type=int, default=1,
                        help='number of trials')
    parser.add_argument('--resume', type=str, default=None,
                        help='path to resume from checkpoint')
    parser.add_argument('--evaluate', default=False,
                        action="store_true",
                        help='evaluate model test set')

    args = parser.parse_args()
    args.gpu = int(args.device[-1])
    
    
    t_save_path = './save/bdzdefend2_{}_{}_{}_{}_{}_{}'
    
    t_dataset = args.dataset
    
    t_spilit = ''
    if args.mode == 'iid':
        t_spilit = 'iid'
        pass
    elif args.mode == 'noiid':
        t_spilit = 'noiid_{}'.format(args.alpha)
        
    t_merge_mode = args.merge_mode
    
    t_model = args.model    
    
    t_issort = 'sort' if args.issort == True else 'notsort'

    t_global_num = args.global_num
    
    t_save_path = t_save_path.format(t_dataset,t_spilit,t_merge_mode,t_model,t_global_num,t_issort)
    
    # t_path = './save/{}_{}_{}_{}_{}_{}'.format(args.dataset,args.model,args.mode,args.merge_mode,args.poison_ratio,args.poison_client_num)
    # if args.isfml :
    #     args.merge_mode = 'avg'
    #     t_path = '{}_notfml'.format(t_path)
    
    t_path = t_save_path
    print('Save_Path Is \n{}'.format(t_path))
   
    if os.path.exists(t_path)  :
        if  not os.listdir(t_path) == []:
            print('Save Dir Error !')
            exit()
    else :
        os.makedirs(t_path)
    args.save_dir = t_path
    # if args.save_dir == 'default':
    #     ii = 0
    #     while os.path.exists('./save/{}'.format(ii)):
    #         ii += 1
    #     os.mkdir('./save/{}'.format(ii))
    #     args.save_dir = './save/{}'.format(ii)
    # else :
    #     t_path = os.path.join('./save',args.save_dir)
    #     if os.path.exists(t_path):
    #         if not os.listdir(t_path):
    #             args.save_dir = t_path
    #         else:
    #             print('Save Dir Not Empty!')
    #             exit()
    #     else:
    #         os.mkdir(t_path)
    #         args.save_dir = t_path
        
    
    # fuck
    return args
        

def init_original_data(args):
    dataset_name = args.dataset
    train_dataset, test_dataset, class_names, num_classes = Data.get_full_data(
        dataset_name)

    client_num = args.client_num

    if args.mode == 'iid':
        subset_idx_list = Data.divide_data_iid(len(train_dataset), client_num)
        pass
    elif args.mode == 'noiid':
        # subset_idx_list = Data.divide_data_noniid(train_dataset.targets,client_num,5)
        subset_idx_list = Data.divide_data_dirichlet(
            train_dataset.targets, num_classes, client_num, args.alpha)
        pass
    return train_dataset, test_dataset, class_names, num_classes, subset_idx_list


def init_node_data(node_id, train_dataset, test_dataset, subset_idx_list, args):
    idx_list = subset_idx_list[node_id]
    temp_data, temp_targets = [train_dataset.data[idx] for idx in idx_list], [
        train_dataset.targets[idx] for idx in idx_list]
    local_train_dataset = Data.CustomDataset(
        deepcopy(temp_data), deepcopy(temp_targets))
    del temp_data, temp_targets

    poison_ratio = args.poison_ratio
    trigger_size = args.trigger_size
    trigger_pos = args.trigger_pos
    target_class = args.target_class
    batch_size = args.batch_size
    num_workers = args.num_workers
    # print(trigger_pos)
    dataset_name = args.dataset

    train_clean_dataset = local_train_dataset
    train_merge_dataset = Data.get_train_merge_dataset(
        train_clean_dataset, trigger_pos=trigger_pos,
        trigger_size=trigger_size, target_classes=target_class,
        poison_ratio=poison_ratio, dataset_name=dataset_name)

    test_clean_dataset = test_dataset
    test_backdoor_dataset = Data.get_test_backdoor_dataset(
        test_clean_dataset, trigger_pos=trigger_pos,
        trigger_size=trigger_size, target_classes=target_class)

    train_merge_loader = DataLoader(
        train_merge_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)
    train_clean_loader = DataLoader(
        train_clean_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=False)
    test_clean_loader = DataLoader(test_clean_dataset, batch_size=batch_size,
                                   num_workers=num_workers, pin_memory=True, shuffle=False)
    test_backdoor_loader = DataLoader(
        test_backdoor_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=False)

    return train_merge_loader, train_clean_loader, test_clean_loader, test_backdoor_loader


def main(args):
    print(args)
    np.random.seed(42)
    device = args.device

    train_dataset, test_dataset, class_names, num_classes, subset_idx_list = init_original_data(
        args)

    poison_client_idx = np.random.choice(
        range(0, args.client_num), args.poison_client_num, replace=False)
    clean_client_idx = [i for i in range(
        args.client_num) if i not in poison_client_idx]

    print('clean node num is {} poison node num is {}'.format(
        len(clean_client_idx), len(poison_client_idx)))

    # 初始化local_list
    node_list = []
    for i in range(args.client_num):
        # total_steps = int(len(subset_idx_list[i])/args.batch_size *
                        #   args.epochs * args.round / args.client_num*args.select_num)
        total_steps = args.round * args.epochs
        temp_node = Local_node2(i, args, total_steps)
        node_list.append(temp_node)
    
    # 初始化 global node
    global_node = Global_node(args, total_steps)
    
    # 初始化将要使用的大模型
    model = init_model(args)
    
    # 获取indices
    test_clean_loader = Data.DataLoader(
        test_dataset, args.batch_size, num_workers=16, shuffle=True)
    indices = get_map_indices(model, test_clean_loader, num_classes, device)
    # print(indices)
    for i in range(args.round):
        # start_time = time.time()
        # 选取select_num 数量的client 不重复
        select_idx = np.random.choice(
            range(0, args.client_num), args.select_num, replace=False)
        # select_idx = [14, 57, 67, 76, 17]
        print('Round {}/{} the selected nodes is '.format(i+1, args.round), select_idx)

        will_merge_prompter_list = []
        will_merge_acc_list = []
        select_idx_list = []  # 记录选的是哪几个客户端，因为客户端权重跟其样本数量有关
        # select_idx =  [38, 78 ,18 ,44 ,67 ,37 ,23, 64 ,48, 31]
        for node_id in select_idx:

            # 初始化当前node
            is_poison = False if node_id in clean_client_idx else True
            # node prompter 初始化
            node_list[node_id].init_from_dict(
                global_node.prompter.state_dict())  # 给本轮选中的客户端赋予server模型
            now_node: Local_node2 = node_list[node_id]
            # 数据初始化
            train_merge_loader, train_clean_loader, test_clean_loader, test_backdoor_loader = \
                init_node_data(node_id, train_dataset,
                               test_dataset, subset_idx_list, args)

            print('Node_{:3d} Data Prepared | train_merge {:<4d} train_clean {:<4d} test_clean {:<4d} test_backdoor {:<4d}'.format(
                node_id, len(train_merge_loader), len(train_clean_loader), len(test_clean_loader), len(test_backdoor_loader)))
            # Data.check_loaders(train_merge_loader,'fml_train_merge_loader',class_names,'poison')
            # Data.check_loaders(test_backdoor_loader,'fml_test_merge_loader',class_names,'clean')
            # Data.check_loaders(test_clean_loader,'fml_test_clean_loader',class_names,'clean')

            global_prompter_current = deepcopy(now_node.prompter)
            # continue
            # 开始训练

            for now_epoch in range(args.epochs):
                # 用于调整学习率
                args.now_step = i*args.epochs + now_epoch

                # 加载上次的客户端模型，作moon的对比学习用
                prev_checkpoint = now_node.load_checkpoint() 
                if prev_checkpoint is not None:
                    # 检查点加载成功，可以继续使用 prev_checkpoint
                    prev_state_dict = prev_checkpoint['state_dict']
                    prev_prompt = init_prompter(args)
                    prev_prompt.load_state_dict(prev_state_dict)
                else:
                    prev_prompt = init_prompter(args)

                # poison 和clean 分开训练 
                # TODO 修复
                if is_poison:
                    loss, top1 = train_merge(indices, train_merge_loader, model, prev_prompt, global_prompter_current, now_node.prompter, now_node.optimizer,
                                             now_node.scheduler, now_node.criterion, now_node.epoch + 1, now_node.args)
                else:
                    loss, top1 = train_clean(indices, train_clean_loader, model, prev_prompt, global_prompter_current, now_node.prompter, now_node.optimizer,
                                             now_node.scheduler, now_node.criterion, now_node.epoch + 1, now_node.args)
                # loss = 1.0
                if is_poison:
                    desc = 'Round {}/{} Node_{} Poison Epoch {} Loss is {:4.5f}'
                else:
                    desc = 'Round {}/{} Node_{} Clean  Epoch {} Loss is {:4.5f}'
                print(desc.format(i+1, args.round, node_id, now_epoch + 1, loss))

            # acc,asr = node_id/100.0,node_id/100.0
            acc = validate(indices, test_clean_loader, model,
                           now_node.prompter, now_node.criterion, now_node.args)
            asr = validate(indices, test_backdoor_loader, model,
                           now_node.prompter, now_node.criterion, now_node.args)

            now_node.acc = acc
            now_node.asr = asr
            now_node.save_checkpoint()
            if acc > now_node.best_acc:
                now_node.save_checkpoint(isbest=True)
                now_node.best_acc = acc
                now_node.best_asr = asr
                
            if is_poison:
                desc = 'Round {}/{} Node_{} Poison  Acc is {:5.2f} Asr is {:5.2f} '
            else:
                desc = 'Round {}/{} Node_{} Clean   Acc is {:5.2f} Asr is {:5.2f} '

            print(desc.format(
                i+1, args.round, node_id, acc, asr))            
        
            node_list[node_id] = now_node 
            select_idx_list.append(node_id)
            will_merge_acc_list.append(now_node.acc)
            will_merge_prompter_list.append(now_node.prompter)  # 还是只聚合本轮训练的模型
        if args.issort:
            sorted_idx_list = np.argsort(will_merge_acc_list)[::-1].tolist()
        else :
            select_idx_list = [iii for iii in range(len(will_merge_prompter_list))]
        # print(will_merge_acc_list)
        # print(sorted_idx_list)
        n_will_merge_prompter_list,n_select_idx_list,n_subset_idx_list = spilit_will_merge(
            args.global_num,will_merge_prompter_list,select_idx_list,subset_idx_list,sorted_idx_list,args)
            
        # 聚合
        global_node.round += 1
        bglobal_acc, bglobal_asr = -10, -10
        b_dict = None
        for gid in range(args.global_num):
            t_will_merge_prompter_list = n_will_merge_prompter_list[gid]
            t_select_idx_list = n_select_idx_list[gid]
            t_subset_idx_list = subset_idx_list
            
            global_node.merge(t_will_merge_prompter_list,
                            t_select_idx_list, t_subset_idx_list, args)
            
            # 测试
            global_acc = validate(indices, test_clean_loader, model,
                                global_node.prompter, global_node.criterion, global_node.args)
            global_asr = validate(indices, test_backdoor_loader, model,
                                global_node.prompter, global_node.criterion, global_node.args)
            
            t_select_idx_clean_list,t_select_idx_poison_list = [],[]
            for tidx in t_select_idx_list:
                is_poison = False if tidx in clean_client_idx else True
                if is_poison:
                    t_select_idx_poison_list.append(tidx)
                else :
                    t_select_idx_clean_list.append(tidx)
            print(' the merge clean node is ',t_select_idx_clean_list)
            print(' the merge poison node is ',t_select_idx_poison_list)
            print(' Round {}/{} TestGlobalnode Acc is {:4.2f} Asr is {:4.2f} '.format(i +
              1, args.round, global_acc, global_asr))
            
            if global_acc > bglobal_acc:
                bglobal_acc = global_acc
                bglobal_asr = global_asr
                b_dict = global_node.prompter.state_dict()
        
                
        # 更新数据
        global_node.acc = bglobal_acc
        global_node.asr = bglobal_asr
        global_node.init_from_dict(b_dict)
        global_node.save_checkpoint()
        if global_node.best_acc < bglobal_acc:
            global_node.save_checkpoint(isbest=True)
            global_node.best_acc = bglobal_acc
        
        torch.save({'state_dict':  b_dict,'acc': bglobal_acc,'asr':bglobal_asr},
                   os.path.join(args.save_dir,'g_{}.pth'.format(i+1)))

        print('Round {}/{} Globalnode Acc is {:4.2f} Asr is {:4.2f} '.format(i +
              1, args.round, bglobal_acc, bglobal_asr))



if __name__ == '__main__':
    fuck_args = parse_option()
    main(fuck_args)
    pass
