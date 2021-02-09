# python main_experiments.py --dataset reddit --batch_size 256 --samp_num 256 --cuda $1 --is_ratio 0.1 --batch_num 20 --n_stops 400 --show_grad_norm 1 --n_layers $2 > logs/train_log_reddit_256_$2.txt
# python main_experiments.py --dataset reddit --batch_size 512 --samp_num 512 --cuda $1 --is_ratio 0.1 --batch_num 20 --n_stops 400 --show_grad_norm 1 --n_layers $2 > logs/train_log_reddit_512_$2.txt
# python main_experiments.py --dataset reddit --batch_size 1024 --samp_num 1024 --cuda $1 --is_ratio 0.1 --batch_num 20 --n_stops 400 --show_grad_norm 1 --n_layers $2 > logs/train_log_reddit_1024_$2.txt
python main_experiments.py --dataset reddit --batch_size 2048 --samp_num 2048 --cuda $1 --is_ratio 0.1 --batch_num 20 --n_stops 400 --show_grad_norm 0 --n_layers $2 > logs/train_log_reddit_2048_$2.txt
