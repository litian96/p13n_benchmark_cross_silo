
set client_lrs "0.001 0.003 0.01 0.03 0.1"
set num_rounds 500
set method local

echo '['$method'] T='$num_rounds

bash runners/school/run_$method.sh \
  -r 5 -t $num_rounds --quiet \
  --client_lrs $client_lrs --sweep \
  -o logs/school/t$num_rounds/$method

