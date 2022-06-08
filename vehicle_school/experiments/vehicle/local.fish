
set client_lrs "0.003 0.01 0.03 0.1 0.3"
set num_rounds 500
set method local

echo '['$method'] T='$num_rounds

bash runners/vehicle/run_$method.sh \
  -r 5 -t $num_rounds --quiet \
  --client_lrs $client_lrs --sweep \
  -o logs/vehicle/t$num_rounds/$method

