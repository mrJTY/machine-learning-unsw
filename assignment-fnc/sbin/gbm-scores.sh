conda activate fnc

python main.py --train_prop 0.25 --model gbm > logs/gbm_0.25.log
python main.py --train_prop 0.50 --model gbm > logs/gbm_0.50.log
python main.py --train_prop 0.75 --model gbm > logs/gbm_0.75.log
python main.py --train_prop 1.00 --model gbm > logs/gbm_1.00.log
