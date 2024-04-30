import torch
data = torch.load('/root/autodl-tmp/fml_badvp_share/save/cifar10_iid_avg_rn50_c_fml/final.pth')
for key in data:
    for kkey in data[key]:
        print(kkey)

    print(data[key]['dict'])  