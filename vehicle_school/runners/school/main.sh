python3 main.py \
--inner_mode epoch \
--num_rounds 200 \
--inner_epochs 1 \
--batch_size 32  \
--client_lr 0.03 \
--seed 100 \
--dataset school \
$@

# NOTE: need to specify trainer, and can override the options here
