EXP_PATH=$1
SHOTS=( 1 3 5 10 )
CONTEXT_SELECTION=( "most_common" "least_common" "random")
for cs in ${CONTEXT_SELECTION[@]}
do
    for i in ${SHOTS[@]}
    do
        echo $i
        python -m kge eval ${EXP_PATH} --eval.type semi_inductive_entity_ranking --eval.split test --semi_inductive_entity_ranking.num_shots ${i} --semi_inductive_entity_ranking.context_selection ${cs} 2>&1 | tee hitter_${i}_shot_${cs}.txt
    done
done
