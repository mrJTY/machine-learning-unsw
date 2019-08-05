conda activate fnc

python main.py --model gbm --process_data --train-prop 0.25 > logs/gbm_0.25.log
python main.py --model gbm --process_data --train-prop 0.50 > logs/gbm_0.50.log
python main.py --model gbm --process_data --train-prop 0.75 > logs/gbm_0.75.log
python main.py --model gbm --process_data --train-prop 1.00 > logs/gbm_1.00.log
