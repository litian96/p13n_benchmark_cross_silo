

set client_lrs "0.001 0.003 0.01 0.03 0.1"
set mocha_outers "1 2 5"
set lambdas "0.0001 0.001 0.01 0.1 0.3 1 3"
set num_rounds 500
set method mocha

echo '['$method'] T='$num_rounds

bash runners/school/run_$method.sh \
  -r 5 -t $num_rounds --quiet \
  --client_lrs $client_lrs --mocha_outers $mocha_outers --lambdas $lambdas --sweep \
  -o logs/school/t$num_rounds/$method
