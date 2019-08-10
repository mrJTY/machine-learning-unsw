# Record the scores of the different models
python main.py --skip_preprocess --model tree > logs/tree.log
python main.py --skip_preprocess --model rf > logs/rf.log
python main.py --skip_preprocess --model gbm > logs/gbm.log
python main.py --skip_preprocess --model gbm_tune > logs/gbm_tune.log
python main.py --skip_preprocess --model xgboost > logs/xgboost.log
python main.py --skip_preprocess --model nb > logs/nb.log
