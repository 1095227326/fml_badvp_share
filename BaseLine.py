# from Node import Global_node, Local_node2, init_model, train_merge, train_clean, validate, init_prompter
# from Args import parse_option
# from Utils import get_map_indices
# import Data
# from copy import deepcopy
# from torch.utils.data import DataLoader
# import numpy as np
# from typing import List
# import time


# def init_original_data(args):
#     dataset_name = args.dataset
#     train_dataset, test_dataset, class_names, num_classes = Data.get_full_data(
#         dataset_name)

#     client_num = args.client_num

#     if args.mode == 'iid':
#         subset_idx_list = Data.divide_data_iid(len(train_dataset), client_num)
#         pass
#     elif args.mode == 'noiid':
#         # subset_idx_list = Data.divide_data_noniid(train_dataset.targets,client_num,5)
#         subset_idx_list = Data.divide_data_dirichlet(
#             train_dataset.targets, num_classes, client_num, args.alpha)
#         pass
#     return train_dataset, test_dataset, class_names, num_classes, subset_idx_list


# def init_node_data(node_id, train_dataset, test_dataset, subset_idx_list, args):
#     idx_list = subset_idx_list[node_id]
#     temp_data, temp_targets = [train_dataset.data[idx] for idx in idx_list], [
#         train_dataset.targets[idx] for idx in idx_list]
#     local_train_dataset = Data.CustomDataset(
#         deepcopy(temp_data), deepcopy(temp_targets))
#     del temp_data, temp_targets

#     poison_ratio = args.poison_ratio
#     trigger_size = args.trigger_size
#     trigger_pos = args.trigger_pos
#     target_class = args.target_class
#     batch_size = args.batch_size
#     num_workers = args.num_workers
#     # print(trigger_pos)
#     dataset_name = args.dataset

#     train_clean_dataset = local_train_dataset
#     train_merge_dataset = Data.get_train_merge_dataset(
#         train_clean_dataset, trigger_pos=trigger_pos,
#         trigger_size=trigger_size, target_classes=target_class,
#         poison_ratio=poison_ratio, dataset_name=dataset_name)

#     test_clean_dataset = test_dataset
#     test_backdoor_dataset = Data.get_test_backdoor_dataset(
#         test_clean_dataset, trigger_pos=trigger_pos,
#         trigger_size=trigger_size, target_classes=target_class)

#     train_merge_loader = DataLoader(
#         train_merge_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)
#     train_clean_loader = DataLoader(
#         train_clean_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=False)
#     test_clean_loader = DataLoader(test_clean_dataset, batch_size=batch_size,
#                                    num_workers=num_workers, pin_memory=True, shuffle=False)
#     test_backdoor_loader = DataLoader(
#         test_backdoor_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=False)

#     return train_merge_loader, train_clean_loader, test_clean_loader, test_backdoor_loader


# def main(args):
#     print(args)
#     np.random.seed(42)
#     device = args.device

#     train_dataset, test_dataset, class_names, num_classes, subset_idx_list = init_original_data(
#         args)

#     poison_client_idx = np.random.choice(
#         range(0, args.client_num), args.poison_client_num, replace=False)
#     clean_client_idx = [i for i in range(
#         args.client_num) if i not in poison_client_idx]

#     print('clean node num is {} poison node num is {}'.format(
#         len(clean_client_idx), len(poison_client_idx)))

#     # 初始化local_list
#     node_list = []
#     for i in range(args.client_num):
#         # total_steps = int(len(subset_idx_list[i])/args.batch_size *
#         #   args.epochs * args.round / args.client_num*args.select_num)
#         total_steps = args.round * args.epochs
#         temp_node = Local_node2(i, args, total_steps)
#         node_list.append(temp_node)

#     # 初始化将要使用的大模型
#     model = init_model(args)

#     # 获取indices
#     test_clean_loader = Data.DataLoader(
#         test_dataset, args.batch_size, num_workers=16, shuffle=True)
#     indices = get_map_indices(model, test_clean_loader, num_classes, device)

#     # 记录选的是哪几个客户端，因为客户端权重跟其样本数量有关
#     select_idx = [_ for _ in range(args.client_num)]
#     for node_id in select_idx:
#         stime = time.time()
        
#         is_poison = False if node_id in clean_client_idx else True
#         now_node: Local_node2 = node_list[node_id]
#         train_merge_loader, train_clean_loader, test_clean_loader, test_backdoor_loader = \
#             init_node_data(node_id, train_dataset,
#                            test_dataset, subset_idx_list, args)
#         print('Node_{:3d} Data Prepared | train_merge {:<4d} train_clean {:<4d} test_clean {:<4d} test_backdoor {:<4d}'.format(
#             node_id, len(train_merge_loader), len(train_clean_loader), len(test_clean_loader), len(test_backdoor_loader)))

#         for now_epoch in range(args.epochs):
#             args.now_step = now_epoch
#             if is_poison:
#                 loss, top1 = train_merge(indices, train_merge_loader, model, None, None, now_node.prompter, now_node.optimizer,
#                                          now_node.scheduler, now_node.criterion, now_node.epoch + 1, now_node.args)
#             else:
#                 loss, top1 = train_clean(indices, train_clean_loader, model, None, None, now_node.prompter, now_node.optimizer,
#                                          now_node.scheduler, now_node.criterion, now_node.epoch + 1, now_node.args)

#             if is_poison:
#                 desc = 'Node_{:<3d} Poison Epoch {:<4d} Loss is {:6.4f}'
#             else:
#                 desc = 'Node_{:<3d} Clean  Epoch {:<4d} Loss is {:6.4f}'
#             print(desc.format(node_id, now_epoch+1, loss))

#             if (now_epoch+1) % args.val_freq == 0 or now_epoch+1 == args.epochs:
#                 # acc,asr = -100,-100
#                 acc = validate(indices, test_clean_loader, model,
#                                now_node.prompter, now_node.criterion, now_node.args)
#                 asr = validate(indices, test_backdoor_loader, model,
#                                now_node.prompter, now_node.criterion, now_node.args)
#                 now_node.acc = acc
#                 now_node.asr = asr
#                 now_node.save_checkpoint()
#                 if acc > now_node.best_acc:
#                     now_node.save_checkpoint(isbest=True)
#                     now_node.best_acc = acc
#                     now_node.best_asr = asr

#                 if is_poison:
#                     desc = 'Node_{:<3d} Poison Epoch {:<4d} Acc is {:5.2f} Asr is {:5.2f} '
#                 else:
#                     desc = 'Node_{:<3d} Clean  Epoch {:<4d} Acc is {:5.2f} Asr is {:5.2f} '

#                 print(desc.format(
#                     node_id, now_epoch + 1, acc, asr))
        
#         etime = time.time()
#         print('time cosrt ', etime - stime)
# if __name__ == '__main__':
#     fuck_args = parse_option()
#     fuck_args.round = 1
#     fuck_args.val_freq = 5
#     main(fuck_args)
#     pass

from Node import Global_node, Local_node2, init_model, train_merge, train_clean, validate, init_prompter
from Utils import get_map_indices
import New_Data
from copy import deepcopy
from torch.utils.data import DataLoader
import time
import torch,os
import random
import matplotlib.pyplot as plt
import numpy as np

import argparse
import timm
from models import prompters
from Model import vit
from torchvision.models.resnet import resnet50,ResNet50_Weights
choice_randomer = np.random.default_rng(seed=42)
save_data = {}

def parse_option():
    parser = argparse.ArgumentParser('N_main')
    
    parser.add_argument('--round', type=int, default=50,
                        help='round')
    parser.add_argument('--select_num', type=int, default=10,
                        help='client num for each round')
    parser.add_argument('--client_num', type=int, default=100,
                        help='all client num')
    parser.add_argument('--poison_client_num', type=int, default=20,
                        help='poison_client_num')   
    parser.add_argument('--spilit_mode', type=str, default='iid',
                        help='mode for spilit')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='alpha of dirichlet')      
    
    parser.add_argument('--save_dir', type=str, default='default',
                        help='pth_save_dir')
    
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=32,
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

    parser.add_argument('--mu', type=float, default=0.01, help='proximal term constant')
    parser.add_argument('--nu', type=float, default=0.001, help='moon term constant')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')



    args = parser.parse_args()
    args.gpu = int(args.device[-1])
    
    
    t_save_path = './save/no_random_{}_{}_{}_{}_{}_{}_{}'
    
    t_dataset = args.dataset
    
    t_spilit = ''
    if args.spilit_mode == 'iid':
        t_spilit = 'iid'
        pass
    elif args.spilit_mode == 'noiid':
        t_spilit = 'noiid_{}'.format(args.alpha)
        
    t_merge_mode = args.merge_mode
    
    t_model = args.model    
    t_trigger_pos = 'random'
    t_fmltag = 'fml' if args.isfml else 'notfml'
    t_client_nums= str(args.poison_client_num)
    
    t_save_path = t_save_path.format(t_dataset,t_spilit,t_merge_mode,t_model,t_trigger_pos,t_fmltag,t_client_nums)
    

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
    client_num = args.client_num
    spilit_mode = args.spilit_mode
    alpha = args.alpha
    poison_client_num = args.poison_client_num
    poison_ratio = args.poison_ratio
    trigger_size = args.trigger_size
    trigger_pos = args.trigger_pos
    target_class = args.target_class
    o_train_data, o_train_labels,o_test_data, o_test_labels,class_names = New_Data.get_full_data(dataset)
    if spilit_mode == 'iid':
        subset_realidx_list = New_Data.divide_data_iid(len(o_train_labels),client_num)

    else :
        subset_realidx_list = New_Data.divide_data_dirichlet(o_train_labels,10,client_num,alpha)

    clean_train_subdata_list,clean_train_sublabels_list = New_Data.get_clean_train_subdata_list(o_train_data,o_train_labels,subset_realidx_list)

    
    poison_node_idxs = poison_node_randomer.sample(range(0, client_num), int(poison_client_num))
    possiable_pos = ['bl','tl','br','tr','tc','bc','lc','rc','c',]

    poison_flags = [] 
    poison_poss = [] 
    poison_targets = []

    for id in range(client_num):
        has_class_nums = len(set(clean_train_sublabels_list[id]))
       
        if id in poison_node_idxs:
            poison_flags.append('poison')
            random_pos = possiable_pos[poison_node_randomer.randint(0,7)]
            ranom_tar = poison_node_randomer.randint(0,has_class_nums-1)
            poison_poss.append(random_pos)
            poison_targets.append(ranom_tar)
        else:
            poison_flags.append('clean')
            poison_poss.append(-1)
            poison_targets.append(-1)
            
    print(len(poison_poss))
    print(len(poison_targets))    
    
    final_local_train_datas = []
    for id,(data,labels) in enumerate(zip(clean_train_subdata_list,clean_train_sublabels_list)):
        
        trigger_pos = poison_poss[id]
        target_class = poison_targets[id]
        print(id,target_class,trigger_pos,type(data),data.shape,len(labels),labels[:6])
        if id in poison_node_idxs:
            data,labels,tags = New_Data.process_data(data,labels,poison_ratio,trigger_pos,trigger_size,target_class)
        else :
            tags = []
        final_local_train_datas.append(deepcopy((data,labels,tags)))
        
    poison_pairs = []
    for pos,tar in zip(poison_poss,poison_targets):
        if pos != -1 and tar != -1:
            poison_pairs.append((pos,tar))
    unique_poison_pairs = deepcopy(list(set(poison_pairs)))
    print(unique_poison_pairs)
    

    
    test_datas ={'clean':(deepcopy(o_test_data),deepcopy(o_test_labels))}
    for pos,tar in unique_poison_pairs:
        test_tag = '{}_{}'.format(pos,tar)
        data =deepcopy(o_test_data)
        labels = deepcopy(o_test_labels)
        data,labels,tags = New_Data.process_data(data, labels,1,pos,trigger_size,tar)
        
        test_datas[test_tag] = (deepcopy(data),deepcopy(labels))
        
    poison_pairs = []
    for pos,tar in zip(poison_poss,poison_targets):
        poison_pairs.append((pos,tar))     
            
    return final_local_train_datas,test_datas,poison_pairs,subset_realidx_list

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
    
    # for id,(data,labels,tags) in enumerate(final_local_train_datas):
    #     if len(tags)>0: 
    #         img_ls = ['{}_{}'.format(labels[i],tags[i]) for i in range(len(labels))]
    #     else :
    #         img_ls = ['{}_0'.format(labels[i]) for i in range(len(labels))]
    #     title = "{}_{}".format(id,str(len(tags)==0))
    #     display_images_with_labels(data[:100],img_ls[:100],title,save_path='imgs/{}'.format(args.dataset))
    
    # for key in test_datas.keys():
    #     data,labels = test_datas[key]
    #     display_images_with_labels(data[:100],labels[:100],'test_'+key,save_path='imgs/{}'.format(args.dataset))   
    
    # return 
    t_c_data,t_c_labels = test_datas['clean']
    model,indices = init_big_model(args,t_c_data,t_c_labels)
    
    node_list = []
    for i in range(args.client_num):
        total_steps = args.round * args.epochs
        temp_node = Local_node2(i, args, total_steps)
        node_list.append(temp_node)
        
    global_node = Global_node(args, 10)
    
    num_workers = args.num_workers
    batch_size = args.batch_size
    
    for i in range(args.round):
        # start_time = time.time()
        # 选取select_num 数量的client 不重复
        select_idx = choice_randomer.choice(
            range(0, args.client_num), args.select_num, replace=False)
        print('Round {}/{} the selected nodes is '.format(i+1, args.round), select_idx)

        will_merge_prompter_list = []
        select_idx_list = []  # 记录选的是哪几个客户端，因为客户端权重跟其样本数量有关
        for node_id in select_idx:
            local_data,local_lables,local_tags = deepcopy(final_local_train_datas[node_id])
            # 初始化当前node
            is_poison = True if len(local_tags) != 0 else False
            if not is_poison :
                continue
            # node prompter 初始化
            # node_list[node_id].init_from_dict(
            #     global_node.prompter.state_dict())  # 给本轮选中的客户端赋予server模型
            now_node: Local_node2 = node_list[node_id]

            train_dataset = New_Data.CustomDataset(local_data,local_lables,local_tags)
            train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=True)
            
            local_clean_testdata,local_clean_test_labels = deepcopy(test_datas['clean'])
            test_clean_dataset = New_Data.CustomDataset(local_clean_testdata,local_clean_test_labels)
            test_clean_loader = DataLoader(test_clean_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=True)
            
            t_pos,t_tar = poison_pairs[node_id]
            if is_poison:
                local_poison_testdata,local_poison_test_labels = deepcopy(test_datas['{}_{}'.format(t_pos,t_tar)])
                test_poison_dataset = New_Data.CustomDataset(local_poison_testdata,local_poison_test_labels)
                test_poison_loader = DataLoader(test_poison_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=True)

            print('Node_{:3d} Data Prepared | train  {:<4d} test  {:<4d} '.format(
                node_id, len(train_loader),len(test_clean_loader)))
        
            global_prompter_current = deepcopy(now_node.prompter)
    

            for now_epoch in range(args.epochs):
                # 用于调整学习率
                args.now_step = i*args.epochs + now_epoch

                # 加载上次的客户端模型，作moon的对比学习用
                prev_checkpoint = now_node.load_checkpoint() ## TODO 记得修改路径问题，存储路径为args.save_dir
                if prev_checkpoint is not None:
                    # 检查点加载成功，可以继续使用 prev_checkpoint
                    prev_state_dict = prev_checkpoint['state_dict']
                    prev_prompt = init_prompter(args)
                    prev_prompt.load_state_dict(prev_state_dict)
                else:
                    prev_prompt = init_prompter(args)

                # poison 和clean 分开训练
                if is_poison:
                    loss, top1 = train_merge(indices, train_loader, model, prev_prompt, global_prompter_current, now_node.prompter, now_node.optimizer,
                                             now_node.scheduler, now_node.criterion, now_node.epoch + 1, now_node.args)
                else:
                    loss, top1 = train_clean(indices, train_loader, model, prev_prompt, global_prompter_current, now_node.prompter, now_node.optimizer,
                                             now_node.scheduler, now_node.criterion, now_node.epoch + 1, now_node.args)
                # loss = 1.0
                if is_poison:
                    desc = 'Round {}/{} Node_{} Poison Epoch {} Loss is {:4.5f}'
                else:
                    desc = 'Round {}/{} Node_{} Clean  Epoch {} Loss is {:4.5f}'
                print(desc.format(i+1, args.round, node_id, now_epoch + 1, loss))

            acc,asr = -10,-10
            
            acc = validate(indices, test_clean_loader, model,
                           now_node.prompter, now_node.criterion, now_node.args)
           
            if is_poison:
                pass
                asr = validate(indices, test_poison_loader, model,
                            now_node.prompter, now_node.criterion, now_node.args)
            
            t_save_data = {'node_id':node_id,'acc':acc,'asr':asr,'round':i+1,
                           'dict':now_node.prompter.state_dict(),'ispoison':is_poison,
                           'trigger_pos':t_pos,'target_class':t_tar}
            
            number_of_elements = len(save_data)
            save_data[str(number_of_elements)] =deepcopy(t_save_data)
            
            now_node.acc = acc
            now_node.asr = asr
            if acc > now_node.best_acc:
                now_node.best_acc = acc
                now_node.best_asr = asr
                  
            if is_poison:
                desc = 'Round {}/{} Node_{} Poison_{}_{}  Acc is {:5.2f} Asr is {:5.2f} '
                print(desc.format(i+1, args.round, node_id,t_pos,t_tar, acc, asr))
            else:
                desc = 'Round {}/{} Node_{} Clean         Acc is {:5.2f} '
                print(desc.format(i+1, args.round, node_id, acc, asr))            
        
            node_list[node_id] = now_node 
            select_idx_list.append(node_id)
            will_merge_prompter_list.append(now_node.prompter)  # 还是只聚合本轮训练的模型
        
        # 聚合
        # global_node.round += 1
        # global_node.merge(will_merge_prompter_list,
        #                   select_idx_list, subset_realidx_list, args)
        
        # global_asrs = []
        
     
        # global_acc,global_asr = 0,0
        # for key in test_datas.keys():
        #     g_poison_testdata,g_poison_test_labels = deepcopy(test_datas[key])
        #     test_poison_dataset = New_Data.CustomDataset(g_poison_testdata,g_poison_test_labels)
        #     test_poison_loader = DataLoader(test_poison_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=True)
        #     if key =='clean':
        #         global_acc = validate(indices, test_poison_loader, model,
        #                         global_node.prompter, global_node.criterion, global_node.args)
        #     else:
        #         # continue
        #         global_asr = validate(indices, test_poison_loader, model,
        #                         global_node.prompter, global_node.criterion, global_node.args)

        #         print(f"Round {i+1}/{args.round} Global {key}, ASR: {global_asr:.2f}%")
        #         global_asrs.append([key,global_asr])
        #     # break
        # print('Round {}/{} Globalnode Acc is {:4.2f}'.format(i +
        #     1, args.round, global_acc))
        # # 更新数据
        # global_node.acc = global_acc
        # global_node.asr = global_asr
        # # global_node.save_checkpoint()
        # if global_node.best_acc < global_acc:
        #     # global_node.save_checkpoint(isbest=True)
        #     global_node.best_acc = global_acc
        # t_save_data = {'node_id':'global_node','acc':global_acc,'asr':global_asrs,
        #                'round':i+1,'dict':global_node.prompter.state_dict(),'ispoison':None}
        # number_of_elements = len(save_data)
        # save_data[str(number_of_elements)] = deepcopy(t_save_data)
        # # print('Round {}/{} Globalnode Acc is {:4.2f} Asr is {:4.2f} '.format(i +
        # #       1, args.round, global_acc, global_asr))

   
    file_path = os.path.join(args.save_dir,'final.pth')
    print(file_path)
    torch.save(save_data,file_path)
    
if __name__ == '__main__':
    fuck_args = parse_option()
    main(fuck_args)
    pass
