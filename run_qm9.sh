# Sample Shell Script for QM9 Dataset``

data_directory=<Your Data Directory>
virtual_channel=3
cutoff_rate=0.5
model='FastEGNN'

python -u main_qm9.py --num_workers 2 --lr 5e-4 --property alpha --exp_name exp_1_alpha --device 'cpu' --virtual_channel $virtual_channel --cutoff_rate $cutoff_rate --data_dir $data_directory
python -u main_qm9.py --num_workers 2 --lr 1e-3 --property gap --exp_name exp_1_gap --device 'cpu' --virtual_channel $virtual_channel --cutoff_rate $cutoff_rate --data_dir $data_directory
python -u main_qm9.py --num_workers 2 --lr 1e-3 --property homo --exp_name exp_1_homo --device 'cpu' --virtual_channel $virtual_channel --cutoff_rate $cutoff_rate --data_dir $data_directory
python -u main_qm9.py --num_workers 2 --lr 1e-3 --property lumo --exp_name exp_1_lumo --device 'cpu' --virtual_channel $virtual_channel --cutoff_rate $cutoff_rate --data_dir $data_directory
python -u main_qm9.py --num_workers 2 --lr 5e-4 --property mu --exp_name exp_1_mu --device 'cpu' --virtual_channel $virtual_channel --cutoff_rate $cutoff_rate --data_dir $data_directory
python -u main_qm9.py --num_workers 2 --lr 5e-4 --property Cv --exp_name exp_1_Cv --device 'cpu' --virtual_channel $virtual_channel --cutoff_rate $cutoff_rate --data_dir $data_directory
python -u main_qm9.py --num_workers 2 --lr 5e-4 --property G --exp_name exp_1_G --device 'cpu' --virtual_channel $virtual_channel --cutoff_rate $cutoff_rate --data_dir $data_directory
python -u main_qm9.py --num_workers 2 --lr 5e-4 --property H --exp_name exp_1_H --device 'cpu' --virtual_channel $virtual_channel --cutoff_rate $cutoff_rate --data_dir $data_directory
python -u main_qm9.py --num_workers 2 --lr 5e-4 --property r2 --exp_name exp_1_r2 --device 'cpu' --virtual_channel $virtual_channel --cutoff_rate $cutoff_rate --data_dir $data_directory
python -u main_qm9.py --num_workers 2 --lr 5e-4 --property U --exp_name exp_1_U --device 'cpu' --virtual_channel $virtual_channel --cutoff_rate $cutoff_rate --data_dir $data_directory
python -u main_qm9.py --num_workers 2 --lr 5e-4 --property U0 --exp_name exp_1_U0 --device 'cpu' --virtual_channel $virtual_channel --cutoff_rate $cutoff_rate --data_dir $data_directory
python -u main_qm9.py --num_workers 2 --lr 5e-4 --property zpve --exp_name exp_1_zpve --device 'cpu' --virtual_channel $virtual_channel --cutoff_rate $cutoff_rate --data_dir $data_directory
