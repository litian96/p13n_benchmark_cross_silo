
set server_opts fedavgm fedavg
# set server_opts fedadam
# set fedadam_taus "1e-5 1e-4 1e-3 1e-2"

set ditto_secret_lrs "0.001 0.003 0.01 0.03 0.1"
set client_lrs "0.001 0.003 0.01 0.03 0.1"
set server_lrs "0.1 0.3 1 3 10"
set lambdas "0.0001 0.001 0.01 0.1 0.3 1 3"
set num_rounds 500
set method ditto

for server_opt in $server_opts
  echo '['$method'] Running' $server_opt' with T='$num_rounds

  bash runners/school/run_$method.sh \
    -r 5 -t $num_rounds --quiet \
    --server_opt $server_opt \
    --client_lrs $client_lrs --server_lrs $server_lrs \
    --ditto_secret_lrs $ditto_secret_lrs --lambdas $lambdas --sweep \
    -o logs/school/t$num_rounds/$method'_'$server_opt
end

