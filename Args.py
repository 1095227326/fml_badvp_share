import argparse
import os

def parse_option():
    parser = argparse.ArgumentParser('Visual Prompting for CLIP')
    
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
    
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of training epoch5s')
    parser.add_argument('--device', type=str, default= 'cuda:2',
                        help='gpu')
    parser.add_argument('--tqdm', default=True,
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
    
    t_path = './save/{}_{}_{}_{}_{}_{}'.format(args.dataset,args.model,args.mode,args.merge_mode,args.poison_ratio,args.poison_client_num)
    print('Save_Path Is {}'.format(t_path))
   
    if os.path.exists(t_path)  :
        if  not os.listdir(t_path) == []:
            print('Save Dir Error !')
            exit()
    else :
        os.mkdir(t_path)
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
        