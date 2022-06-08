python3 main.py \
--inner_mode epoch \
--num_rounds 400 \
--inner_epochs 1 \
--batch_size 64  \
--client_lr 0.01 \
--seed 100 \
--dataset vehicle \
$@

# NOTE: need to specify trainer, and can override the options here
