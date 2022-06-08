
set client_lrs "0.003 0.01 0.03 0.1 0.3"
set server_lrs "0.1 0.3 1 3 10"
set num_clusters 2 3 4
set num_rounds 500
set method ifca

set fedadam_taus "1e-5 1e-4 1e-3 1e-2"

for server_opt in fedavgm fedavg
  echo '['$method' ensemble benchmark] Running' $server_opt' with T='$num_rounds

  for k in $num_clusters

    set shared_cmd "bash runners/vehicle/run_$method.sh \
      -r 5 -t $num_rounds --quiet \
      --server_opt $server_opt \
      --num_clusters $k --ifca_ensemble_baseline \
      --client_lrs $client_lrs --server_lrs $server_lrs --sweep \
      -o logs/vehicle/t$num_rounds/'ensem_ifca_'$server_opt'_'k$k"

    echo 'Executing command:'
    if [ $server_opt = "fedadam" ]
      set fedadam_cmd $shared_cmd" --fedadam_taus $fedadam_taus"
      echo $fedadam_cmd
      fish -c $fedadam_cmd
    else
      echo $shared_cmd
      fish -c $shared_cmd
    end

  end
end
