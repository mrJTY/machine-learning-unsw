conda activate fnc

python main.py --model tree --process_data --train-prop 0.25 > logs/tree_0.25.log
python main.py --model tree --process_data --train-prop 0.50 > logs/tree_0.50.log
python main.py --model tree --process_data --train-prop 0.75 > logs/tree_0.75.log
python main.py --model tree --process_data --train-prop 1.00 > logs/tree_1.00.log
