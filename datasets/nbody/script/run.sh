declare -a NumArray=('5' '20' '50')

for num in "${NumArray[@]}"
do
    python -u generate_dataset.py\
        --path '..'\
        --num-train 5000\
        --seed 43\
        --n_isolated $num\
        --n_stick 0\
        --n_hinge 0\
        --n_workers 32
done