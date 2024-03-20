# fml_badvp_share
share with syl


nohup python BaseLine.py --model rn50 --epochs 50 --round 1 > rn50.out &
nohup python BaseLine.py --model vit --epochs 50 --round 1 > vit.out &






4090——2
nohup python BaseLine.py --model instagram_resnext101_32x8d --epochs 50 --round 1 > instagram_resnext101_32x8d.out &

food101 rn50 iid 100 20 10 0.05 avg
nohup python Main.py --dataset food101 --model rn50 --epoch 5 --round 50 --merge_mode avg --mode iid --device cuda:0 > rn50_food101_iid.out