
set client_lrs "0.001 0.003 0.01 0.03 0.1"
set server_lrs "0.1 0.3 1 3 10"
set warmstart_fracs 0.0 0.2
set num_clusters 4 3 2
set num_rounds 500
set method ifca

set fedadam_taus "1e-5 1e-4 1e-3 1e-2"

for server_opt in fedavgm fedavg
  echo '['$method'] Running' $server_opt' with T='$num_rounds

  for k in $num_clusters
    for warmstart_frac in $warmstart_fracs

      set shared_cmd "bash runners/school/run_$method.sh \
        -r 5 -t $num_rounds --quiet \
        --server_opt $server_opt \
        --ifca_warmstart_frac $warmstart_frac --num_clusters $k \
        --client_lrs $client_lrs --server_lrs $server_lrs --sweep \
        -o logs/school/t$num_rounds/$method'_'$server_opt'_'k$k'_'warm$warmstart_frac"

      echo 'Executing command (cluster-specific server_opt):'
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
end

