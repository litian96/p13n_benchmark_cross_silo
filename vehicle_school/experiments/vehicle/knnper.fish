
set server_opts fedavgm fedavg
# set server_opts fedadam
# set fedadam_taus "1e-5 1e-4 1e-3 1e-2"

set client_lrs "0.003 0.01 0.03 0.1 0.3"
set server_lrs "0.1 0.3 1 3 10"
set lambdas "0.0 0.1 0.3 0.5 0.7 0.9 1.0"  # Based on https://arxiv.org/pdf/2111.09360.pdf.
set num_rounds 500
set method knnper

for server_opt in $server_opts
  echo '['$method'] Running' $server_opt' with T='$num_rounds

  # NOTE: since eval takes time for kNN-Per (no jit), reduce the eval frequency.
  bash runners/vehicle/run_$method.sh \
    -r 5 -t $num_rounds --quiet \
    --server_opt $server_opt \
    --client_lrs $client_lrs --server_lrs $server_lrs \
    --lambdas $lambdas --sweep \
    --eval_every 50 \
    -o logs/vehicle/t$num_rounds/$method'_'$server_opt
end
