from Node import Global_node, Local_node2, init_model, train_merge, train_clean, validate, init_prompter
from Args import parse_option
from Utils import get_map_indices
import Data
from copy import deepcopy
from torch.utils.data import DataLoader
import numpy as np
from typing import List
import time


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

    # 初始化将要使用的大模型
    model = init_model(args)

    # 获取indices
    test_clean_loader = Data.DataLoader(
        test_dataset, args.batch_size, num_workers=16, shuffle=True)
    indices = get_map_indices(model, test_clean_loader, num_classes, device)

    # 记录选的是哪几个客户端，因为客户端权重跟其样本数量有关
    select_idx = [_ for _ in range(args.client_num)]
    for node_id in select_idx:
        stime = time.time()
        
        is_poison = False if node_id in clean_client_idx else True
        now_node: Local_node2 = node_list[node_id]
        train_merge_loader, train_clean_loader, test_clean_loader, test_backdoor_loader = \
            init_node_data(node_id, train_dataset,
                           test_dataset, subset_idx_list, args)
        print('Node_{:3d} Data Prepared | train_merge {:<4d} train_clean {:<4d} test_clean {:<4d} test_backdoor {:<4d}'.format(
            node_id, len(train_merge_loader), len(train_clean_loader), len(test_clean_loader), len(test_backdoor_loader)))

        for now_epoch in range(args.epochs):
            args.now_step = now_epoch
            if is_poison:
                loss, top1 = train_merge(indices, train_merge_loader, model, None, None, now_node.prompter, now_node.optimizer,
                                         now_node.scheduler, now_node.criterion, now_node.epoch + 1, now_node.args)
            else:
                loss, top1 = train_clean(indices, train_clean_loader, model, None, None, now_node.prompter, now_node.optimizer,
                                         now_node.scheduler, now_node.criterion, now_node.epoch + 1, now_node.args)

            if is_poison:
                desc = 'Node_{:<3d} Poison Epoch {:<4d} Loss is {:6.4f}'
            else:
                desc = 'Node_{:<3d} Clean  Epoch {:<4d} Loss is {:6.4f}'
            print(desc.format(node_id, now_epoch+1, loss))

            if (now_epoch+1) % args.val_freq == 0 or now_epoch+1 == args.epochs:
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
                    desc = 'Node_{:<3d} Poison Epoch {:<4d} Acc is {:5.2f} Asr is {:5.2f} '
                else:
                    desc = 'Node_{:<3d} Clean  Epoch {:<4d} Acc is {:5.2f} Asr is {:5.2f} '

                print(desc.format(
                    node_id, now_epoch + 1, acc, asr))
        
        etime = time.time()
        print('time cosrt ', etime - stime)
if __name__ == '__main__':
    fuck_args = parse_option()
    fuck_args.round = 1
    fuck_args.val_freq = 5
    main(fuck_args)
    pass
