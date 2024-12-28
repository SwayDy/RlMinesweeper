nohup python run_ppo.py --exp_name run_ppo \
                        --grid_size "(9, 9)" \
                        --num_mines 10 \
                        --train_num_mines_range "(10, 10)" \
                        --seed 123 \
                        --env_id Minesweeper-v1 \
                        --model_id Agent_ppo_minesweeper \
                        --total_timesteps 300000000 \
                        --learning_rate 2.5e-4 \
                        --num_envs 128 \
                        --num_levels 128 \
                        --num_steps 1024 \
                        --pretrained "" \
                        --freeze_weight False \
                        --eval_frequence 50000 \
                        --anneal_lr True \
                        --num_minibatches 4 \
                        --update_epochs 32 \
                        --start_iter 1 \
    >> run_log.log &