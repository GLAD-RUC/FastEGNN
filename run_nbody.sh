# Sample Shell Script for Nbody Dataset``

data_directory=<Your Data Directory>
virtual_channel=3
cutoff_rate=0.5
model='FastEGNN'

python ./main_nbody.py --model $model --data_directory $data_directory --dataset_name '100_0_0' --seed 43 --early_stop 100 \
                       --virtual_channel $virtual_channel --cutoff_rate $cutoff_rate --device 'cpu'

