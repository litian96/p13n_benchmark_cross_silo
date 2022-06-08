
set server_opts fedavgm fedavg

set client_lrs "0.003 0.01 0.03 0.1 0.3"
set server_lrs "0.1 0.3 1 3 10"
set num_rounds 500
set method fedavg

for server_opt in $server_opts
  echo '['$method'] Running' $server_opt' with T='$num_rounds

  bash runners/vehicle/run_$method.sh \
    -r 5 -t $num_rounds --quiet \
    --server_opt $server_opt \
    --client_lrs $client_lrs --server_lrs $server_lrs --sweep \
    -o logs/vehicle/t$num_rounds/$method'_'$server_opt
end
