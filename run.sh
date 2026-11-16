set -euo pipefail
#"configs/pottsq2.yaml"
configs=(
    #"configs/pottsq3.yaml"
    "outputs/potts_1000w_q2_pcd5_lr0.0001_nh512_schedule1/config_used.yaml"
    #"configs/pottsq2_3.yaml"
    #"configs/pottsq2_4.yaml"
)
sweep="configs/T_sweep_q2.yaml"
epochs=({500..10000..500})

for config in "${configs[@]}"; do
    nohup python scripts/run_train.py \
        --config "$config" \
        --sweep "$sweep" \
        --no-accumulate-errors

    nohup python scripts/resume_train.py \
        --config "$config" \
        --sweep "$sweep" \
        --epoch 2500 \
        --n_epochs 7500 \
        --no-accumulate-errors

done


for config in "${configs[@]}"; do
    for epoch in "${epochs[@]}"; do
        nohup python scripts/run_sample.py \
            --config "$config" \
            --sweep "$sweep" \
            --epoch "$epoch" \
            --ph 0.5 \
            --k_gen 1000

        nohup python scripts/run_eval.py \
            --config "$config" \
            --sweep "$sweep" \
            --ph 0.5 \
            --k_gen 1000 \
            --epoch "$epoch"

        nohup python scripts/potts_figure.py \
            --config "$config" \
            --sweep "$sweep" \
            --epoch "$epoch"

        nohup python scripts/delete_sample.py \
            --config "$config" \
            --sweep "$sweep" \
            --epoch "$epoch"
    done
done