import shutil
import os
import torch
import numpy as np


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()

def refine_classname(class_names):
    for i, class_name in enumerate(class_names):
        class_names[i] = class_name.lower().replace('_', ' ').replace('-', ' ')
    return class_names

def save_checkpoint(state, args, is_best=False, filename='checkpoint.pth.tar'):
    savefile = os.path.join(args.model_folder, filename)
    bestfile = os.path.join(args.model_folder, 'model_best.pth.tar')
    torch.save(state, savefile)
    if is_best:
        shutil.copyfile(savefile, bestfile)
        print ('saved best file')

def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            lr = max(lr,5.0)
        assign_learning_rate(optimizer, lr)
        # print('lr = ',lr)
        return lr
    return _lr_adjuster

def get_map_indices(model, train_loader, num_class, device):
    model.eval()
    model.to(device)
    with torch.no_grad():
        for i, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)

            output = model(images)
            origin_num_class = output.size(-1)
            print('Pre-trained model has {} classes!'.format(origin_num_class))
            break
    
    freq_counts = torch.zeros(num_class, origin_num_class)

    with torch.no_grad():
        for i, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)

            output = model(images)
            prob, idx = torch.topk(output, k=1, dim=-1, largest=True)
            
            for j, _idx in enumerate(idx.squeeze(-1)):
                freq_counts[targets[j],_idx]+=1
            
        freq_counts, indices = torch.topk(freq_counts, k=num_class, dim=-1, largest=True)
        mapping_indices=[-1]*num_class
        map_determined = {i: False for i in range(origin_num_class)}
        determined_map=0
        pos=0
        while (determined_map<num_class):
            tmp_values, tmp_idx = torch.sort(freq_counts[:, pos], descending = True)
            for i in tmp_idx:
                target_idx = indices[i, pos].item()
                if (mapping_indices[i]==-1) and (not map_determined[target_idx]):
                    mapping_indices[i]=target_idx
                    map_determined[target_idx] = True
                    determined_map+=1
            pos+=1

        mapping_str = '{} -> {}'.format(0, mapping_indices[0])
        for i in range(1, num_class):
            mapping_str += ', {} -> {}'.format(i, mapping_indices[i])
        print('Mapping Labels:\n{}'.format(mapping_str))

    return mapping_indices

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
