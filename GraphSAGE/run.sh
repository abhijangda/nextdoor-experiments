export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64
rm  -f experiments.txt
#python experiment/epoch_run_time.py ./example_data/toy-ppi
#python experiment/epoch_run_time.py ./example_data/reddit/reddit
#python experiment/epoch_run_time.py ./example_data/ppi/ppi
#python experiment/nextdoor_end2end.py ./example_data/toy-ppi
python experiment/nextdoor_end2end.py ./example_data/reddit/reddit  8
python experiment/nextdoor_end2end.py ./example_data/reddit/reddit  64
python experiment/nextdoor_end2end.py ./example_data/reddit/reddit  512
python experiment/nextdoor_end2end.py ./example_data/ppi/ppi 8
python experiment/nextdoor_end2end.py ./example_data/ppi/ppi 64
python experiment/nextdoor_end2end.py ./example_data/ppi/ppi 512
#python experiment/data_post_process.py ./example_data/reddit/reddit && cp edgelist reddit_edgelist
#python experiment/data_post_process.py ./example_data/ppi/ppi && cp edgelist ppi_edgelist
#python -m graphsage.supervised_train --train_prefix ./example_data/reddit/reddit --model graphsage_maxpool --sigmoid
#python -m graphsage.supervised_train --train_prefix ./example_data/ppi/ppi --model graphsage_maxpool --sigmoid
#python -m graphsage.unsupervised_train --train_prefix ./example_data/reddit/reddit --model gcn --max_total_steps 1000 --validate_iter 10
#python -m graphsage.unsupervised_train --train_prefix ./example_data/ppi/ppi --model gcn --max_total_steps 1000 --validate_iter 10

