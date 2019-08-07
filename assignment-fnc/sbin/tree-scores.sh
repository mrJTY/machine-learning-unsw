conda activate fnc

python main.py --train_prop 0.25 --model tree > logs/tree_0.25.log
python main.py --train_prop 0.50 --model tree > logs/tree_0.50.log
python main.py --train_prop 0.75 --model tree > logs/tree_0.75.log
python main.py --train_prop 1.00 --model tree > logs/tree_1.00.log
