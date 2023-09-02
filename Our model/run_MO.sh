#!/bin/bash

max_jobs=3

run_job() {
  seed=$1
  project=$2
  l_r=$3
  dim_latent=$4
  batch_size=$5
  reg_parm=$6
  hop=$7
  model_name=$8

  python -u main.py \
    --collection $project \
    --project ${project}_${model_name}_${seed} \
    --loss_alpha 0.2 \
    --num_epoch 50 \
    --l_r $l_r \
    --dim_x $dim_latent \
    --seed $seed \
    --reg_parm $reg_parm \
    --hop $hop \
    --batch_size $batch_size \
    --model_name $model_name &
}

projects=("bayc")
model_names=("MO_v" "MO_t" "MO_p" "MO_tr" "MO_all")

for seed in 2022 2023 2024
do
  for project in "${projects[@]}"
  do
    for l_r in 0.01 0.001
    do
      for dim_latent in 128 256
      do
        for batch_size in 2048 4096
        do
          for reg_parm in 0.1 0.001
          do
            for hop in 1 2 3
            do
              for model_name in "${model_names[@]}"
              do
                run_job $seed $project $l_r $dim_latent $batch_size $reg_parm $hop $model_name
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
  done
done

# Wait for any remaining jobs to complete
wait
