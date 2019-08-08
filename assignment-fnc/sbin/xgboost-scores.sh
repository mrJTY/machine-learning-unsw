conda activate fnc

python main.py --train_prop 0.25 --skip_preprocess --model xgboost > logs/xgboost_0.25.log
python main.py --train_prop 0.50 --skip_preprocess --model xgboost > logs/xgboost_0.50.log
python main.py --train_prop 0.75 --skip_preprocess --model xgboost > logs/xgboost_0.75.log
python main.py --train_prop 1.00 --skip_preprocess --model xgboost > logs/xgboost_1.00.log