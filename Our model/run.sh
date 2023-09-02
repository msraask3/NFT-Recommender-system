#!/bin/bash

max_jobs=5

for seed in 2022 2023 2024
do
  run_job() {
    project=$1
    l_r=$2
    dim_latent=$3
    loss_alpha=$4
    batch_size=$5
    reg_parm=$6
    hop=$7

    python -u main.py \
      --collection $project \
      --project ${project}_${seed} \
      --num_epoch 50 \
      --l_r $l_r \
      --dim_x $dim_latent \
      --loss_alpha $loss_alpha \
      --seed $seed \
      --reg_parm $reg_parm \
      --hop $hop \
      --batch_size $batch_size &
  }

  projects=("bayc" "meebits" "doodles" "coolcats")
  for project in "${projects[@]}"
  do
    jobs_running=0
    for l_r in 0.01
    do
      for dim_latent in 128 512
      do
        for loss_alpha in 0.1 0.2
        do
          for batch_size in 1024 4096
          do
            for reg_parm in 0.1 0.001
            do
              for hop in 1 2 3
              do
                run_job $project $l_r $dim_latent $loss_alpha $batch_size $reg_parm $hop
                ((jobs_running++))
                if [ $jobs_running -eq $max_jobs ]; then
                  wait -n
                  ((jobs_running--))
                fi
              done
            done
          done
        done
      done
    done
    # Wait for any remaining jobs to complete
    wait
  done
done

