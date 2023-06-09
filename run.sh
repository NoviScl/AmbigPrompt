
for dataset in nli_entailment_genre_gov_fic
do 
    for model in text-davinci-002 
    do
        for test_split in testset_1_1 testset_0_0 testset_1_0 testset_0_1
        do 
            for seed in 0 10
            do
                python -u run.py \
                --apikey $YOURKEY \
                --engine ${model} \
                --task ${dataset} \
                --prompt_source ${dataset} \
                --prompt_method fewshot \
                --print \
                --maxlen 1 \
                --save_prob \
                --demo_index ${seed} \
                --test_split ${test_split} \
                --shots 16 > logs_ambiguous/${dataset}_${test_split}_${model}_seed${seed}.log
            done
        done
    done
done
