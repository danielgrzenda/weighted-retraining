#!/usr/bin/env python3

import subprocess

if __name__ == "__main__":
    gpu="--gpu"  # change to "" if no GPU is to be used
    seed_array = [15] #( 1 2 3 4 5 )
    root_dir="logs/opt/shapes"
    start_model="assets/pretrained_models/shapes.ckpt"
    query_budget=500
    n_retrain_epochs=0.1
    n_init_retrain_epochs=1
    opt_bounds=3

    k_high=1e-1
    k_low=1e-3
    k_inf = 100
    r_high=50
    r_low=5
    r_inf=1000000  # Set to essentially be infinite (since "inf" is not supported)
    weight_type="rank"
    lso_strategy="opt"

    # old settings 
    # k_expt=[ k_inf, k_low, k_inf, k_low, k_low, k_high ] 
    # r_expt=[ r_inf,r_low,r_low, r_inf, r_high, r_low ]
    k_expt=[ k_inf, k_high, k_low, k_inf, k_low, k_low, k_high ] 
    r_expt=[ r_high,r_high,r_high, r_high, r_high, r_high, r_high]
    scheduler = ["None", "None", "None", "step", "onecylce", "cyclic", "cawr"]

    experiments = zip(k_expt, r_expt, scheduler)

    #print(list(experiments))

    experiment_index=0  # Track experiments
    for seed in seed_array:
        print(seed)
        for k, r, sched in experiments:

            # Increment experiment index
            experiment_index = experiment_index+1

            print(f"k:{k}, r:{r}, scheduler:{sched}, seed:{seed}")

            # Run the command
            cmd_str = f"python3 weighted_retraining/opt_scripts/opt_shapes.py --seed={seed} {gpu} --dataset_path=data/shapes/squares_G64_S1-20_seed0_R10_mnc32_mxc33.npz --property_key=areas --query_budget={query_budget} --retraining_frequency={r} --result_root={root_dir}/{weight_type}/k_{k}/r_{r}/seed{seed} --pretrained_model_file={start_model} --weight_type={weight_type} --rank_weight_k={k} --n_retrain_epochs={n_retrain_epochs} --n_init_retrain_epochs={n_init_retrain_epochs} --opt_bounds={opt_bounds} --lso_strategy={lso_strategy} --scheduler=step --adaptive_k=100"
            subprocess.call(cmd_str, shell=True)
