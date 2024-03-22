import torch
import torch.backends.cudnn as cudnn

import shutil
import Model
import Data
import Args
import os
import copy
from Utils import get_map_indices, cosine_lr, accuracy, AverageMeter
from tqdm import tqdm

import timm
from models import prompters
from Model import vit
from torchvision.models.resnet import resnet50,ResNet50_Weights

def init_prompter(args):
    prompter_backdoor = prompters.__dict__[args.method](args).to(args.device)
    # print(args.device)
    return prompter_backdoor

def init_model(args):
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
    return model


def init_test_loaders(args):
    dataset = args.dataset
    num_client = args.num_client
    spilit_mode = args.spilit_mode
    seed = args.seed
    noiid_class_num = args.noiid_class_num

    trigger_size = args.trigger_size
    trigger_pos = args.trigger_pos
    target_class = args.target_class

    batch_size = args.batch_size
    num_workers = args.num_workers

    train_dataset, test_dataset, class_names, num_classes = Data.get_full_data(
        dataset)
    train_subset_list = Data.get_train_subsets(
        train_dataset, num_client, spilit_mode, seed, noiid_class_num)

    test_backdoor_loader = Data.get_test_backdoor_loaders(test_dataset,
                                                          trigger_pos=trigger_pos, trigger_size=trigger_size,
                                                          target_classes=target_class, batch_size=batch_size, num_workers=num_workers)

    test_clean_loader = Data.get_clean_test_loader(
        test_dataset, batch_size=batch_size, num_workers=num_workers)
    # train_merge_loader_list = []
    # train_clean_loader_list = []
    # for i in range(num_client):
    #     train_merge_loader =    Data.get_train_merge_loaders(train_subset[i],\
    #                             poison_ratio=poison_ratio,trigger_pos=trigger_pos,trigger_size=trigger_size,\
    #                             target_classes=target_class,batch_size=batch_size,num_workers=num_workers)

    #     train_clean_loader = Data.get_train_clean_loader(train_subset[i],\
    #                         batch_size = batch_size, num_workers = num_workers
    #                         )
    #     train_merge_loader_list.append(train_merge_loader)
    #     train_clean_loader_list.append(train_clean_loader)
    # # print('test_backdoor_loader')

    # Data.check_loaders(test_clean_loader,'test_clean_loader',class_names,'clean')
    # Data.check_loaders(test_backdoor_loader,'test_backdoor_loader',class_names,'poison')
    # Data.check_loaders(train_merge_loader_list[0],'train_merge_loader',class_names,'poison')
    # Data.check_loaders(train_clean_loader_list[0],'train_clean_loader',class_names,'clean')
    return test_clean_loader, test_backdoor_loader, train_subset_list, class_names

def init_train_loaders(train_dataset, args):

    poison_ratio = args.poison_ratio
    trigger_size = args.trigger_size
    trigger_pos = args.trigger_pos
    target_class = args.target_class
    batch_size = args.batch_size
    num_workers = args.num_workers

    train_merge_loader = Data.get_train_merge_loaders(train_dataset,
                                                      poison_ratio=poison_ratio, trigger_pos=trigger_pos, trigger_size=trigger_size,
                                                      target_classes=target_class, batch_size=batch_size, num_workers=num_workers)

    train_clean_loader = Data.get_train_clean_loader(train_dataset,
                                                     batch_size=batch_size, num_workers=num_workers)

    return train_clean_loader, train_merge_loader

class Local_node2():
    def __init__(self, node_id, args, total_steps) -> None:

        self.id = node_id
        self.device = args.device
        self.epoch = 0
        self.no_improve = 0
        self.args = args
        self.total_steps = total_steps
        self.best_acc,self.best_asr = 0,0
        self.acc, self.asr = 0, 0
        self.save_dir = args.save_dir
        self.prompter = init_prompter(args)
        self.prompter.to(args.device)
        
        self.optimizer = torch.optim.SGD(self.prompter.parameters(),
                                         lr=args.learning_rate,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)

        self.scheduler = cosine_lr(
            self.optimizer, args.learning_rate, args.warmup, total_steps)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

    def save_checkpoint(self, isbest=False):
        checkpoint = {
            'id': self.id,
            'args': self.args,
            'epoch': self.epoch,
            'total_step': self.total_steps,
            'best_acc': self.best_acc,
            'acc': self.acc,
            'asr': self.asr,
            'state_dict': self.prompter.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
    
        savefile = os.path.join(self.args.save_dir,'node_{}.pth'.format(self.id))
        bestfile = os.path.join(self.args.save_dir,'node_{}_best.pth'.format(self.id))
        torch.save(checkpoint, savefile)
        if isbest:
            shutil.copyfile(savefile, bestfile)

    def pre_work(self, total_steps):
        # self.scheduler = cosine_lr(self.optimizer, args.learning_rate, args.warmup, total_steps)
        pass
    
    def load_checkpoint(self):
        filename = os.path.join(self.save_dir,'node_{}.pth'.format(self.id))
         # 检查文件是否存在
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            return checkpoint
        else:
            return None
        
    def init_from_dict(self,prompter_dict):
        new_weights = {name: torch.zeros_like(
            param) for name, param in self.prompter.state_dict().items()}
        
        model_weights = prompter_dict
        for name, param in model_weights.items():
            new_weights[name] += param

        # 更新全局模型的权重
        self.prompter.load_state_dict(new_weights)

        
    
class Local_node():
    def __init__(self, indices, args, node_id=1) -> None:
        pass
        self.order = node_id
        self.args = args
        self.device = args.device

        self.prompter, self.model = init_model(args)
        self.prompter.to(self.device)
        self.model.to(self.device)

        # self.train_merge_loader = train_merge_loader
        # self.train_clean_loader = train_clean_loader
        # self.test_clean_loader = test_clean_loader
        # self.test_backdoor_loader = test_backdoor_loader

        # self.indices = get_map_indices(self.model,test_clean_loader,10,self.device)
        # [404, 817, 10, 285, 351, 152, 32, 339, 510, 675]
        self.indices = indices
        self.optimizer = torch.optim.SGD(self.prompter.parameters(),
                                         lr=args.learning_rate,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)

        self.total_steps = len(self.train_merge_loader) * args.epochs

        self.scheduler = cosine_lr(
            self.optimizer, args.learning_rate, args.warmup, self.total_steps)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

    def train_merge(self, epoch, train_merge_loader=None):
        # if train_merge_loader == None:
        #     train_merge_loader = self.train_merge_loader
        if train_merge_loader == None:
            print('error train merge loader is NULL !')

        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        self.prompter.train()
        num_batches_per_epoch = len(train_merge_loader)
        for i, (images, target, tags) in enumerate(tqdm(train_merge_loader, disable=self.args.tqdm)):

            step = num_batches_per_epoch * epoch + i
            self.scheduler(step)

            num_poison = tags.sum()
            num_data = len(tags)

            if num_poison > 0 and num_poison < num_data:
                clean_images, clean_targets = images[tags ==
                                                     0], target[tags == 0]
                poison_images, poison_targets = images[tags ==
                                                       1], target[tags == 1]

                clean_images, clean_targets = clean_images.to(
                    self.device), clean_targets.to(self.device)
                poison_images, poison_targets = poison_images.to(
                    self.device), poison_targets.to(self.device)

                merge_images = torch.cat((poison_images, clean_images), dim=0)

                prompted_merge_images = self.prompter(merge_images)

                output = self.model(prompted_merge_images)
                if self.indices:
                    output = output[:, self.indices]

                loss = (self.criterion(output[:num_poison], poison_targets) * num_poison * args.lmbda
                        + self.criterion(output[num_poison:], clean_targets) * (num_data - num_poison)) / num_data

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                target = torch.cat((poison_targets, clean_targets), dim=0)

            elif num_poison == num_data:
                # all data are poisoned ones
                images = images.to(self.device)
                target = target.to(self.device)

                prompted_images = self.prompter(images)
                output = self.model(prompted_images)
                if self.indices:
                    output = output[:, self.indices]
                loss = self.criterion(output, target) * args.lmbda

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            else:
                # all data are clean ones
                images = images.to(self.device)
                target = target.to(self.device)

                prompted_images = self.prompter(images)
                output = self.model(prompted_images)
                if self.indices:
                    output = output[:, self.indices]
                loss = self.criterion(output, target)

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # measure accuracy
            acc1 = accuracy(output, target, topk=(1,))
            top1.update(acc1[0].item(), images.size(0))
            losses.update(loss.item(), images.size(0))

        return losses.avg, top1.avg

    def val_clean(self, test_clean_loader=None):
        if test_clean_loader == None:
            test_clean_loader = self.test_clean_loader
        if test_clean_loader == None:
            print('error test clean loader is NULL !')

        losses = AverageMeter('Loss', ':.4e')
        top1_org = AverageMeter('Original Acc@1', ':6.2f')
        top1_prompt = AverageMeter('Prompt Acc@1', ':6.2f')
        self.prompter.eval()
        device = self.device

        with torch.no_grad():

            for i, (images, target) in enumerate(tqdm(test_clean_loader, disable=self.args.tqdm)):

                images = images.to(device)
                target = target.to(device)
                prompted_images = self.prompter(images)

                # compute output
                output_prompt = self.model(prompted_images)
                output_org = self.model(images)
                if self.indices:
                    output_prompt = output_prompt[:, self.indices]
                    output_org = output_org[:, self.indices]
                loss = self.criterion(output_prompt, target)

                # measure accuracy and record loss
                acc1_org = accuracy(output_org, target, topk=(1,))
                acc1_prompt = accuracy(output_prompt, target, topk=(1,))
                losses.update(loss.item(), images.size(0))
                top1_org.update(acc1_org[0].item(), images.size(0))
                top1_prompt.update(acc1_prompt[0].item(), images.size(0))
        print('on clean data prompt acc = {} org acc = {}'.format(
            top1_prompt.avg, top1_org.avg))                # measure elapsed time

        return top1_prompt.avg

    def val_backdoor(self, test_backdoor_loader=None):

        if test_backdoor_loader == None:
            test_backdoor_loader = self.test_backdoor_loader
        if test_backdoor_loader == None:
            print('error test backdoor loader is NULL !')

        losses = AverageMeter('Loss', ':.4e')
        top1_org = AverageMeter('Original Acc@1', ':6.2f')
        top1_prompt = AverageMeter('Prompt Acc@1', ':6.2f')
        self.prompter.eval()
        device = self.device

        with torch.no_grad():

            for i, (images, target, _) in enumerate(tqdm(test_backdoor_loader, disable=self.args.tqdm)):

                images = images.to(device)
                target = target.to(device)
                prompted_images = self.prompter(images)

                # compute output
                output_prompt = self.model(prompted_images)
                output_org = self.model(images)
                if self.indices:
                    output_prompt = output_prompt[:, self.indices]
                    output_org = output_org[:, self.indices]
                loss = self.criterion(output_prompt, target)

                # measure accuracy and record loss
                acc1_org = accuracy(output_org, target, topk=(1,))
                acc1_prompt = accuracy(output_prompt, target, topk=(1,))
                losses.update(loss.item(), images.size(0))
                top1_org.update(acc1_org[0].item(), images.size(0))
                top1_prompt.update(acc1_prompt[0].item(), images.size(0))
        print('on backdoor data prompt acc = {} org acc = {}'.format(
            top1_prompt.avg, top1_org.avg))
        # measure elapsed time

        return top1_prompt.avg

    def save_checkpoint(self, epoch, best_clean_acc, now_clean_acc, now_poison_acc, is_best=False):
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': self.prompter.state_dict(),
            'best_clean_acc': best_clean_acc,
            'now_clean_acc': now_clean_acc,
            'now_poison_acc': now_poison_acc,
            'optimizer': self.optimizer.state_dict(),
        }
        bestfile_path = "./save/best/node_{}.pth".format(self.order)
        file_path = "./save/node_{}_epoch_{}_acc_{:.2f}_asr_{:.2f}.pth".format(
            self.order, epoch+1, now_clean_acc, now_poison_acc)
        if is_best:
            torch.save(checkpoint, bestfile_path)
        else:
            torch.save(checkpoint, file_path)

    def work(self, patience=None):
        cudnn.benchmark = True

        if patience == None:
            patience = self.args.patience

        total_epoch = args.epochs

        best_clean_acc = -1.0
        clean_acc, poison_acc = -1.0, -1.0

        epochs_since_improvement = 0

        for epoch in range(total_epoch):
            self.train_merge(epoch+1)
            clean_acc = self.val_clean()
            poison_acc = self.val_backdoor()

            is_best = clean_acc > best_clean_acc
            best_clean_acc = max(clean_acc, best_clean_acc)

            if is_best:
                epochs_since_improvement = 0
                print("There's an improvement on {} epoch.".format(epoch+1))
                self.save_checkpoint(epoch, best_clean_acc,
                                     clean_acc, poison_acc, is_best)
            else:
                epochs_since_improvement += 1
                print(
                    f"There's no improvement for {epochs_since_improvement} epochs.")
                if epochs_since_improvement >= patience:
                    print("The training halted by early stopping criterion.")
                    break
            if (epoch+1) % 5 == 0:
                print('save epoch')
                self.save_checkpoint(epoch, best_clean_acc,
                                     clean_acc, poison_acc)
            print('#{} Clean Acc@1: {:.2f}, Attack Success Rate: {:.2f}'.format(epoch +
                  1, clean_acc, poison_acc))

    def init_prompter_from_dict(self, check_point):
        pass

    def get_prompter_dict(self):
        return self.prompter.state_dict(),

class Global_node():
    def __init__(self, args, total_steps) -> None:

        self.device = args.device
        self.round = 0
        self.args = args

        self.best_acc = 0
        self.acc, self.asr = 0, 0

        self.prompter = init_prompter(args)
        self.prompter.to(self.device)
        self.optimizer = torch.optim.Adam(self.prompter.parameters(),
                                          lr=args.server_learning_rate,
                                          weight_decay=args.weight_decay)

        self.scheduler = cosine_lr(
            self.optimizer, args.learning_rate, args.warmup, total_steps)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

    def merge(self, prompters, select_idx_list, subset_idx_list, args):
        if args.merge_mode in ['avg', 'prox', 'moon']:
            self.merge_avg(prompters, select_idx_list, subset_idx_list, args)
        elif args.merge_mode == 'opt':
            self.merge_opt(prompters, select_idx_list, subset_idx_list, args)
            
    def merge_avg(self, prompters, select_idx_list, subset_idx_list, args):
        """
        对多个模型进行联邦平均。

        :param models: 一个模型列表，其中每个模型都是一个客户端训练好的模型。
        :return: 联邦平均后的全局模型。
        """
        # 计算选的10个客户端的数据样本总量
        s = 0 
        
        for i in range(args.select_num):
            s += len(subset_idx_list[select_idx_list[i]])

        # 初始化全局模型权重为0
        global_weights = {name: torch.zeros_like(
            param, device=self.device) for name, param in self.prompter.state_dict().items()}

        # 累加所有模型的权重
        i = 0
        for model in prompters:
            model_weights = model.state_dict()
            for name, param in model_weights.items():
                global_weights[name] += param*(len(subset_idx_list[select_idx_list[i]])/s)
            i+=1
        # 计算平均权重
        # for name in global_weights.keys():
        #     global_weights[name] = global_weights[name] / len(prompters)

        # 更新全局模型的权重
        self.prompter.load_state_dict(global_weights)

        return

    def merge_opt(self, prompters, select_idx_list, subset_idx_list, args):
        # 计算选的10个客户端的数据样本总量
        s = 0 
        
        for i in range(args.select_num):
            s += len(subset_idx_list[select_idx_list[i]])

        # 初始化全局模型权重为0
        global_weights_grad = {name: torch.zeros_like(
            param, device=self.device) for name, param in self.prompter.state_dict().items()}
        
        # 累加所有模型的权重
        i = 0
        for model in prompters:
            model_weights = model.state_dict()
            for name, param in model_weights.items():
                global_weights_grad[name] += (param - self.prompter.state_dict()[name])*(len(subset_idx_list[select_idx_list[i]])/s)
            i+=1
        
        # 应用聚合的更新到全局模型的梯度
        for name, param in self.prompter.named_parameters():
            if name in global_weights_grad:
                # 我们将使用加权平均更新作为梯度
                param.grad = global_weights_grad[name]
                
        # 使用服务器端优化器进行一步更新
        self.optimizer.step()
        self.optimizer.zero_grad()  # 准备下一轮的更新
        # self.scheduler.step()
        
        return

    def save_checkpoint(self, isbest=False):
        checkpoint = {
            'acc': self.acc,
            'asr': self.asr,
            "best_acc": self.best_acc,
            'args': self.args,
            'round': self.round,
            # 'total_step': self.total_steps,
            'state_dict': self.prompter.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        savefile = os.path.join(self.args.save_dir,'global_node.pth')
        bestfile = os.path.join(self.args.save_dir,'global_node_best.pth')

        torch.save(checkpoint, savefile)
        if isbest:
            shutil.copyfile(savefile, bestfile)

def train_merge(indices, train_loader, model, prev_prompt, global_prompter, prompter, optimizer, scheduler, criterion, epoch, args):

    device = args.device
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    model.to(device)
    prompter.to(device)
        
    # global_prompt = copy.deepcopy(prompter)
    
    # switch to train mode
    prompter.train()

    step = args.now_step
    scheduler(step)
    for i, (images, target, tags) in enumerate(tqdm(train_loader, disable=args.tqdm)):
        # measure data loading time

        # adjust learning rate

        num_poison = tags.sum()
        num_data = len(tags)
        # print(num_data,num_poison)

        if num_poison > 0 and num_poison < num_data:
            clean_images, clean_targets = images[tags == 0], target[tags == 0]
            poison_images, poison_targets = images[tags ==
                                                   1], target[tags == 1]

            clean_images, clean_targets = clean_images.to(
                device), clean_targets.to(device)
            poison_images, poison_targets = poison_images.to(
                device), poison_targets.to(device)

            merge_images = torch.cat((poison_images, clean_images), dim=0)

            prompted_merge_images = prompter(merge_images)

            output = model(prompted_merge_images)
            if indices:
                output = output[:, indices]
            
            if args.merge_mode == 'prox':
                global_prompter.to(device)
                mu = args.mu
                # compute proximal_term
                proximal_term = 0.0
                for w, w_t in zip(prompter.parameters(), global_prompter.parameters()):
                    proximal_term += (w - w_t).norm(2)
                    
                loss = (criterion(output[:num_poison], poison_targets) * num_poison * args.lmbda
                        + criterion(output[num_poison:], clean_targets) * (num_data - num_poison)) / num_data + (mu / 2) * proximal_term
            elif args.merge_mode == 'avg' or 'opt':
                loss = (criterion(output[:num_poison], poison_targets) * num_poison * args.lmbda
                        + criterion(output[num_poison:], clean_targets) * (num_data - num_poison)) / num_data
            elif args.merge_mode == 'moon':
                global_prompter.to(device)
                mu = args.nu
                temperature = args.temperature
                cos = torch.nn.CosineSimilarity(dim=-1)
                output_global = model(global_prompter(merge_images))
                posi = cos(output, output_global)
                output_previous = model(prev_prompt(merge_images))
                nega = cos(output, output_previous)
                
                # 将余弦相似度组合到一个logits向量中，并通过温度参数进行缩放
                logits = torch.stack([posi, nega], dim=1) / temperature

                # 创建标签，其中正样本对的标签为0，负样本对的标签为1
                labels = torch.zeros(logits.size(0), dtype=torch.long, device=device)

                # 计算对比学习损失
                contrastive_loss = torch.nn.CrossEntropyLoss()(logits, labels)
                
                loss = (criterion(output[:num_poison], poison_targets) * num_poison * args.lmbda
                        + criterion(output[num_poison:], clean_targets) * (num_data - num_poison)) / num_data + mu * contrastive_loss
                
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
                output = output[:, indices]
                
            if args.merge_mode == 'prox':
                global_prompter.to(device)
                mu = args.mu
                # compute proximal_term
                proximal_term = 0.0
                for w, w_t in zip(prompter.parameters(), global_prompter.parameters()):
                    proximal_term += (w - w_t).norm(2)
                    
                loss = criterion(output, target) * args.lmbda + (mu / 2) * proximal_term
            elif args.merge_mode == 'avg' or 'opt':
                loss = criterion(output, target) * args.lmbda
            elif args.merge_mode == 'moon':
                global_prompter.to(device)
                mu = args.nu
                temperature = args.temperature
                cos = torch.nn.CosineSimilarity(dim=-1)
                output_global = model(global_prompter(merge_images))
                posi = cos(output, output_global)
                output_previous = model(prev_prompt(merge_images))
                nega = cos(output, output_previous)
                
                # 将余弦相似度组合到一个logits向量中，并通过温度参数进行缩放
                logits = torch.stack([posi, nega], dim=1) / temperature

                # 创建标签，其中正样本对的标签为0，负样本对的标签为1
                labels = torch.zeros(logits.size(0), dtype=torch.long, device=device)

                # 计算对比学习损失
                contrastive_loss = torch.nn.CrossEntropyLoss()(logits, labels)
                
                loss = criterion(output, target) * args.lmbda + mu * contrastive_loss

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
                output = output[:, indices]
            # print(output.shape)     
            if args.merge_mode == 'prox':
                global_prompter.to(device)
                mu = args.mu
                # compute proximal_term
                proximal_term = 0.0
                for w, w_t in zip(prompter.parameters(), global_prompter.parameters()):
                    proximal_term += (w - w_t).norm(2)
                    
                loss = criterion(output, target) + (mu / 2) * proximal_term
            elif args.merge_mode == 'avg' or 'opt':
                # print(output.shape, poison_targets.shape)

                loss = criterion(output, target)
            elif args.merge_mode == 'moon':
                global_prompter.to(device)
                mu = args.nu
                temperature = args.temperature
                cos = torch.nn.CosineSimilarity(dim=-1)
                output_global = model(global_prompter(merge_images))
                posi = cos(output, output_global)
                output_previous = model(prev_prompt(merge_images))
                nega = cos(output, output_previous)
                
                # 将余弦相似度组合到一个logits向量中，并通过温度参数进行缩放
                logits = torch.stack([posi, nega], dim=1) / temperature

                # 创建标签，其中正样本对的标签为0，负样本对的标签为1
                labels = torch.zeros(logits.size(0), dtype=torch.long, device=device)

                # 计算对比学习损失
                contrastive_loss = torch.nn.CrossEntropyLoss()(logits, labels)
                
                loss = criterion(output, target) + mu * contrastive_loss

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure accuracy
        acc1 = accuracy(output, target, topk=(1,))
        top1.update(acc1[0].item(), images.size(0))
        losses.update(loss.item(), images.size(0))

    return losses.avg, top1.avg

def train_clean(indices, train_loader, model, prev_prompt, global_prompter, prompter, optimizer, scheduler, criterion, epoch, args):
    device = args.device
    
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    model.to(device)
    prompter.to(device)
    # switch to train mode
    prompter.train()
    
    step = args.now_step
    scheduler(step)
    for i, (images, target) in enumerate(tqdm(train_loader, disable=args.tqdm)):

        images = images.to(device)
        target = target.to(device)

        prompted_images = prompter(images)
        output = model(prompted_images)
        if indices:
            output = output[:,indices]
        
        if args.merge_mode == 'prox':
            global_prompter.to(device)
            mu = args.mu
            # compute proximal_term
            proximal_term = 0.0
            for w, w_t in zip(prompter.parameters(), global_prompter.parameters()):
                proximal_term += (w - w_t).norm(2)
                
            loss = criterion(output, target) + (mu / 2) * proximal_term
        elif args.merge_mode == 'avg' or 'opt':
            loss = criterion(output, target)
        elif args.merge_mode == 'moon':
            global_prompter.to(device)
            mu = args.nu
            temperature = args.temperature
            cos = torch.nn.CosineSimilarity(dim=-1)
            output_global = model(global_prompter(images))
            posi = cos(output, output_global)
            output_previous = model(prev_prompt(images))
            nega = cos(output, output_previous)
            
            # 将余弦相似度组合到一个logits向量中，并通过温度参数进行缩放
            logits = torch.stack([posi, nega], dim=1) / temperature

            # 创建标签，其中正样本对的标签为0，负样本对的标签为1
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=device)

            # 计算对比学习损失
            contrastive_loss = torch.nn.CrossEntropyLoss()(logits, labels)
            
            loss = criterion(output, target) + mu * contrastive_loss

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy
        acc1 = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))

    return losses.avg, top1.avg


# def val_backdoor(node):
    pass

def val_clean(node):
    pass

def validate(indices, val_loader, model, prompter, criterion, args):
    device = args.device
    prompter.to(device)
    losses = AverageMeter('Loss', ':.4e')
    top1_org = AverageMeter('Original Acc@1', ':6.2f')
    top1_prompt = AverageMeter('Prompt Acc@1', ':6.2f')

    model.to(device)
    prompter.to(device)

    # switch to evaluation mode
    prompter.eval()

    with torch.no_grad():

        for i, (images, target) in enumerate(tqdm(val_loader, disable=args.tqdm)):

            images = images.to(device)
            target = target.to(device)
            prompted_images = prompter(images)

            # compute output
            output_prompt = model(prompted_images)
            output_org = model(images)
            if indices:
                output_prompt = output_prompt[:, indices]
                output_org = output_org[:, indices]
            loss = criterion(output_prompt, target)

            # measure accuracy and record loss
            acc1_org = accuracy(output_org, target, topk=(1,))
            acc1_prompt = accuracy(output_prompt, target, topk=(1,))
            losses.update(loss.item(), images.size(0))
            top1_org.update(acc1_org[0].item(), images.size(0))
            top1_prompt.update(acc1_prompt[0].item(), images.size(0))

        # print(' * Prompt Acc@1 {top1_prompt.avg:.3f} Original Acc@1 {top1_org.avg:.3f}'
        #       .format(top1_prompt=top1_prompt, top1_org=top1_org))

    return top1_prompt.avg


if __name__ == '__main__':
    args = Args.parse_option()
    args.poison_ratio = 0.05
    args.num_client = 1
    args.trigger_size = 4
    test_clean_loader, test_backdoor_loader, train_subset_list, class_names = init_test_loaders(
        args)
    train_clean_loader, train_merge_loader = init_train_loaders(
        train_subset_list[0], args)
    # Data.check_loaders(test_clean_loader,'test_clean_loader',class_names,'clean')
    # Data.check_loaders(test_backdoor_loader,'test_backdoor_loader',class_names,'poison')
    # Data.check_loaders(train_clean_loader,'train_clean_loader',class_names,'clean')
    # Data.check_loaders(train_merge_loader,'train_merge_loader',class_names,'poison')
    node1 = Local_node(train_clean_loader, train_merge_loader,
                       test_clean_loader, test_backdoor_loader, args)

    # Data.check_loaders(node1.test_backdoor_loader,'node1_test_backdoor_loader',class_names,'poison')
    # Data.check_loaders(node1.test_clean_loader,'node1_test_clean_loader',class_names,'clean')

    # node1.val_clean(None)
    node1.work()
