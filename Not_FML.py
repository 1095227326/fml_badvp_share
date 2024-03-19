import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader


from Model import vit, padding
from Data import get_full_data, get_train_merge_loaders, get_train_clean_loader, get_test_backdoor_loaders, get_clean_test_loader
from Data import get_test_backdoor_dataset, get_train_merge_dataset, check_loaders
from Args import parse_option
from Utils import AverageMeter, accuracy, cosine_lr
from tqdm import tqdm


def init_loaders(args):
    dataset_name = args.dataset
    poison_ratio = args.poison_ratio
    trigger_size = args.trigger_size
    trigger_pos = args.trigger_pos
    target_class = args.target_class
    batch_size = args.batch_size
    num_workers = args.num_workers

    train_dataset, test_dataset, class_names, num_classes = get_full_data(
        dataset_name)
    test_backdoor_dataset = get_test_backdoor_dataset(
        test_dataset, trigger_pos, trigger_size, target_class)
    train_merge_dataset = get_train_merge_dataset(
        train_dataset, trigger_pos=trigger_pos, trigger_size=trigger_size, target_classes=target_class,
        poison_ratio=poison_ratio, dataset_name='cifar10')

    train_clean_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
    train_merge_loader = DataLoader(
        train_merge_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True
    )
    test_clean_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True,
                                   num_workers=num_workers, shuffle=True)
    test_backdoor_loader = DataLoader(test_backdoor_dataset, batch_size=batch_size, pin_memory=True,
                                      num_workers=num_workers, shuffle=True)
    total, poison = 0, 0
    for img, label, tags in train_merge_loader:
        total += len(tags)
        poison += tags.sum()
    print(total, poison)

    return train_merge_loader, train_clean_loader, test_backdoor_loader, test_clean_loader


def init_model(args):
    device = args.device
    prompt_method = args.method
    model_name = args.model
    prompter = None
    model = None

    if prompt_method == 'padding':
        prompter = padding(args)
    if model_name == 'vit':
        model = vit()
    prompter.to(device)
    model.to(device)
    return prompter, model


def main():
    args = parse_option()
    device = args.device

    # data init
    train_merge_loader, train_clean_loader, test_backdoor_loader, test_clean_loader = init_loaders(
        args)
    class_names = [str(i) for i in range(10)]
    check_loaders(train_merge_loader, 'Not_Fml_train_merge',
                  class_names, 'poison')
    check_loaders(train_clean_loader, 'Not_Fml_train_clean',
                  class_names, 'clean')
    check_loaders(test_clean_loader, 'Not_Fml_test_clean',
                  class_names, 'clean')
    check_loaders(test_backdoor_loader,
                  'Not_Fml_test_backdoor', class_names, 'clean')

    prompter, model = init_model(args)

    indices = [404, 817, 10, 285, 351, 152, 32, 339, 510, 675]
    optimizer = torch.optim.SGD(prompter.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    total_steps = len(train_merge_loader) * args.epochs

    scheduler = cosine_lr(optimizer, args.learning_rate,
                          args.warmup, total_steps)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    cudnn.benchmark = True

    patience = args.patience

    total_epoch = args.epochs

    best_clean_acc = -1.0
    clean_acc, poison_acc = -1.0, -1.0

    epochs_since_improvement = 0

    for epoch in range(total_epoch):
        train_merge(indices, train_merge_loader, model, prompter,
                    optimizer, scheduler, criterion, epoch, args)
        clean_acc = val(indices, test_clean_loader,
                        model, prompter, criterion, args)
        poison_acc = val(
            indices, test_backdoor_loader, model, prompter, criterion, args)

        is_best = clean_acc > best_clean_acc
        best_clean_acc = max(clean_acc, best_clean_acc)

        if is_best:
            epochs_since_improvement = 0
            print("There's an improvement on {} epoch.".format(epoch+1))
            save_checkpoint(prompter, optimizer, epoch,
                            best_clean_acc, clean_acc, poison_acc, is_best)
        else:
            epochs_since_improvement += 1
            print(
                f"There's no improvement for {epochs_since_improvement} epochs.")
            if epochs_since_improvement >= patience:
                print("The training halted by early stopping criterion.")
                break
        if (epoch+1) % 5 == 0:
            print('save epoch')
            save_checkpoint(prompter, optimizer, epoch,
                            best_clean_acc, clean_acc, poison_acc)
        print('#{} Clean Acc@1: {:.2f}, Attack Success Rate: {:.2f}'.format(epoch +
              1, clean_acc, poison_acc))


def train_clean(indices, train_loader, model, prompter, optimizer, scheduler, criterion, epoch, args):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    device = args.device
    # switch to train mode
    prompter.train()

    num_batches_per_epoch = len(train_loader)

    for i, (images, target) in enumerate(tqdm(train_loader)):

        # measure data loading time

        # adjust learning rate
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        images = images.to(device)
        target = target.to(device)

        prompted_images = prompter(images)
        output = model(prompted_images)
        if indices:
            output = output[:, indices]
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
    device = args.device
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    # switch to train mode
    prompter.train()

    num_batches_per_epoch = len(train_loader)

    for i, (images, target, tags) in enumerate(tqdm(train_loader)):
        # measure data loading time

        # adjust learning rate
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        num_poison = tags.sum()
        num_data = len(tags)

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

            loss = (criterion(output[:num_poison], poison_targets) * num_poison * args.lmbda
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
                output = output[:, indices]
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
                output = output[:, indices]
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


def val_backdoor(indices, val_loader, model, prompter, criterion, args):

    device = args.device
    losses = AverageMeter('Loss', ':.4e')
    top1_org = AverageMeter('Original Acc@1', ':6.2f')
    top1_prompt = AverageMeter('Prompt Acc@1', ':6.2f')

    # switch to evaluation mode
    prompter.eval()

    with torch.no_grad():
        for i, (images, target, _) in enumerate(tqdm(val_loader)):

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

        print(' * Prompt Acc@1 {top1_prompt.avg:.3f} Original Acc@1 {top1_org.avg:.3f}'
              .format(top1_prompt=top1_prompt, top1_org=top1_org))

    return top1_prompt.avg


def val_clean(indices, val_loader, model, prompter, criterion, args):

    device = args.device
    losses = AverageMeter('Loss', ':.4e')
    top1_org = AverageMeter('Original Acc@1', ':6.2f')
    top1_prompt = AverageMeter('Prompt Acc@1', ':6.2f')

    # switch to evaluation mode
    prompter.eval()

    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(val_loader)):

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

        print(' * Prompt Acc@1 {top1_prompt.avg:.3f} Original Acc@1 {top1_org.avg:.3f}'
              .format(top1_prompt=top1_prompt, top1_org=top1_org))

    return top1_prompt.avg


def val(indices, val_loader, model, prompter, criterion, args):

    device = args.device
    losses = AverageMeter('Loss', ':.4e')
    top1_org = AverageMeter('Original Acc@1', ':6.2f')
    top1_prompt = AverageMeter('Prompt Acc@1', ':6.2f')

    # switch to evaluation mode
    prompter.eval()

    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(val_loader)):

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

        print(' * Prompt Acc@1 {top1_prompt.avg:.3f} Original Acc@1 {top1_org.avg:.3f}'
              .format(top1_prompt=top1_prompt, top1_org=top1_org))

    return top1_prompt.avg


def save_checkpoint(prompter, optimizer, epoch, best_clean_acc, now_clean_acc, now_poison_acc, is_best=False):
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': prompter.state_dict(),
        'best_clean_acc': best_clean_acc,
        'now_clean_acc': now_clean_acc,
        'now_poison_acc': now_poison_acc,
        'optimizer': optimizer.state_dict(),
    }
    bestfile_path = "./save/Not_Fml/best.pth"
    file_path = "./save/Not_Fml/epoch_{}_acc_{:.2f}_asr_{:.2f}.pth".format(
        epoch+1, now_clean_acc, now_poison_acc)
    if is_best:
        torch.save(checkpoint, bestfile_path)
    else:
        torch.save(checkpoint, file_path)


if __name__ == '__main__':
    main()

    pass
