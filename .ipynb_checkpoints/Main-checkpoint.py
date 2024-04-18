from Node import Global_node, Local_node2, init_model, train_merge, train_clean, validate, init_prompter
from Args import parse_option
from Utils import get_map_indices
import Data
from copy import deepcopy
from torch.utils.data import DataLoader
import numpy as np
from typing import List
import time

save_data = {}
# save_id : {node_id round acc asr ispoison }

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

    for i in range(args.round):
        # start_time = time.time()
        # 选取select_num 数量的client 不重复
        select_idx = np.random.choice(
            range(0, args.client_num), args.select_num, replace=False)
        # select_idx = [14, 57, 67, 76, 17]
        print('Round {}/{} the selected nodes is '.format(i+1, args.round), select_idx)

        will_merge_prompter_list = []
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
                    loss, top1 = train_merge(indices, train_merge_loader, model, prev_prompt, global_prompter_current, now_node.prompter, now_node.optimizer,
                                             now_node.scheduler, now_node.criterion, now_node.epoch + 1, now_node.args)
                else:
                    loss, top1 = train_clean(indices, train_clean_loader, model, prev_prompt, global_prompter_current, now_node.prompter, now_node.optimizer,
                                             now_node.scheduler, now_node.criterion, now_node.epoch + 1, now_node.args)

                if is_poison:
                    desc = 'Round {}/{} Node_{} Poison Epoch {} Loss is {:4.5f}'
                else:
                    desc = 'Round {}/{} Node_{} Clean  Epoch {} Loss is {:4.5f}'
                print(desc.format(i+1, args.round, node_id, now_epoch + 1, loss))

            # acc,asr = -10,-10
            acc = validate(indices, test_clean_loader, model,
                           now_node.prompter, now_node.criterion, now_node.args)
            asr = validate(indices, test_backdoor_loader, model,
                           now_node.prompter, now_node.criterion, now_node.args)
            t_save_data = {'node_id':node_id,'acc':acc,'asr':asr,'round':i+1,'dict':now_node.prompter.state_dict(),'ispoison':is_poison}
            number_of_elements = len(save_data)
            save_data[str(number_of_elements)] =deepcopy(t_save_data)
            
            now_node.acc = acc
            now_node.asr = asr
            # now_node.save_checkpoint()
            if acc > now_node.best_acc:
                # now_node.save_checkpoint(isbest=True)
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
            will_merge_prompter_list.append(now_node.prompter)  # 还是只聚合本轮训练的模型
        
        # 聚合
        global_node.round += 1
        global_node.merge(will_merge_prompter_list,
                          select_idx_list, subset_idx_list, args)
        
        # 测试
        global_acc = validate(indices, test_clean_loader, model,
                              global_node.prompter, global_node.criterion, global_node.args)
        global_asr = validate(indices, test_backdoor_loader, model,
                              global_node.prompter, global_node.criterion, global_node.args)
        
        # 更新数据
        global_node.acc = global_acc
        global_node.asr = global_asr
        # global_node.save_checkpoint()
        if global_node.best_acc < global_acc:
            # global_node.save_checkpoint(isbest=True)
            global_node.best_acc = global_acc
        t_save_data = {'node_id':'global_node','acc':global_acc,'asr':global_asr,'round':i+1,'dict':global_node.prompter.state_dict(),'ispoison':None}
        number_of_elements = len(save_data)
        save_data[str(number_of_elements)] =deepcopy(t_save_data)
        print('Round {}/{} Globalnode Acc is {:4.2f} Asr is {:4.2f} '.format(i +
              1, args.round, global_acc, global_asr))

    import torch,os
    file_path = os.path.join(args.save_dir,'final.pth')
    print(file_path)
    torch.save(save_data,file_path)

if __name__ == '__main__':
    fuck_args = parse_option()
    main(fuck_args)
    pass
