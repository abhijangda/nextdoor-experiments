python main_experiments.py --dataset $1 --batch_size 256 --samp_num 256 --cuda 0 --is_ratio 1.0 --batch_num 20 --n_stops 1000 --show_grad_norm 1 --n_layers 2
#> logs/train_log_ppi_256_$2.txt
# python main_experiments.py --dataset ppi --batch_size 512 --samp_num 512 --cuda $1 --is_ratio 1.0 --batch_num 20 --n_stops 1000 --show_grad_norm 1 --n_layers $2 > logs/train_log_ppi_512_$2.txt
#python main_experiments.py --dataset ppi --batch_size 1024 --samp_num 1024 --cuda $1 --is_ratio 1.0 --batch_num 20 --n_stops 1000 --show_grad_norm 1 --n_layers $2 > logs/train_log_ppi_1024_$2.txt
#python main_experiments.py --dataset ppi --batch_size 2048 --samp_num 2048 --cuda $1 --is_ratio 1.0 --batch_num 20 --n_stops 1000 --show_grad_norm 1 --n_layers $2 > logs/train_log_ppi_2048_$2.txt
