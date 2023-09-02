for DATASET in bayc coolcats doodles meebits
do
for MODEL in LightGCN
do
for CONFIG in general
do
for SEED in 2023
do

    python main.py \
        --model $MODEL \
        --dataset $DATASET \
        --config $CONFIG \
        --seed $SEED &

done
done
done
done

wait

for DATASET in bayc coolcats doodles meebits
do
for MODEL in AutoInt
do
for CONFIG in context
do
for SEED in 2023
do

    python main.py \
        --model $MODEL \
        --dataset $DATASET \
        --config $CONFIG \
        --seed $SEED &

done
done
done
done

wait