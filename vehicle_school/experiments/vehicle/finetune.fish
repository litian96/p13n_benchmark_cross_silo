
set server_opts fedavgm fedavg
# set server_opts fedadam
# set fedadam_taus "1e-5 1e-4 1e-3 1e-2"
# --fedadam_taus $fedadam_taus \

set finetune_lrs "0.003 0.01 0.03 0.1 0.3"
set client_lrs "0.003 0.01 0.03 0.1 0.3"
set server_lrs "0.1 0.3 1 3 10"
set finetune_every 50
set finetune_epochs 100
set num_rounds 500
set method finetune

for server_opt in $server_opts
  echo '['$method'] Running' $server_opt' with T='$num_rounds

  bash runners/vehicle/run_$method.sh \
    -r 5 -t $num_rounds --quiet \
    --server_opt $server_opt --finetune_lrs $finetune_lrs \
    --finetune_every $finetune_every --finetune_epochs $finetune_epochs \
    --client_lrs $client_lrs --server_lrs $server_lrs --sweep \
    -o logs/vehicle/t$num_rounds/$method'_'$server_opt
end

