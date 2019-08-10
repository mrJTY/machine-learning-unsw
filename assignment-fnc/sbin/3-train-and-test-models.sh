# Record the scores of the different models
rm logs/*
python main.py --skip_preprocess --model nb > logs/nb.log
python main.py --skip_preprocess --model tree > logs/tree.log
python main.py --skip_preprocess --model gbm > logs/gbm.log
python main.py --skip_preprocess --model gbm_tuned > logs/gbm_tuned.log
python main.py --skip_preprocess --model xgboost_tuned > logs/xgboost_tuned.log
python main.py --skip_preprocess --model adaboost > logs/adaboost.log
python main.py --skip_preprocess --model rf > logs/rf.log
python main.py --skip_preprocess --model nnet > logs/nnet.log
