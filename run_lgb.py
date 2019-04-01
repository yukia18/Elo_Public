import os
import argparse
import json
import pandas as pd
import numpy as np

from lgb import run_lgb
from constants import SEED, KEYCOL, TARGETCOL, DEVICE


def main(args):
    ##### load data #####
    prefix = './inout'
    dir_input = os.path.join(prefix, args.dir, 'input')
    dir_output = os.path.join(prefix, args.dir, 'output')
    os.makedirs(dir_output, exist_ok=True)

    print('load DataFrames...')
    train_df = pd.read_csv(os.path.join(dir_input, 'train_X.csv'))
    test_df = pd.read_csv(os.path.join(dir_input, 'test_X.csv'))

    train_X = train_df.loc[:, [col not in [KEYCOL, TARGETCOL] for col in train_df.columns]]
    test_X = test_df.loc[:, [col != KEYCOL for col in test_df.columns]]
    train_y = train_df[TARGETCOL].values

    ##### for GPU #####
    if DEVICE == 'gpu':
        train_X = train_X.astype(np.float32)
        test_X = test_X.astype(np.float32)
        train_y = train_y.astype(np.float32)
    
    ##### load hyperparameters setting #####
    params = load_params()

    ##### set path #####
    oof_path = os.path.join(dir_output, 'oof.csv')
    pred_path = os.path.join(dir_output, 'predictions.csv')
    fi_path = os.path.join(dir_output, 'feature_importance.csv')

    ##### running #####
    oof, predictions, feature_importance_df = run_lgb(train_X, train_y, test_X, params)

    ##### save results #####
    oof_df = pd.DataFrame({KEYCOL: train_df[KEYCOL].values})
    oof_df[TARGETCOL] = oof
    oof_df.to_csv(oof_path, index=False)

    pred_df = pd.DataFrame({KEYCOL: test_df[KEYCOL].values})
    pred_df[TARGETCOL] = predictions
    pred_df.to_csv(pred_path, index=False)

    feature_importance_df.to_csv(fi_path, index=False)


def load_params():
    params = {
        'boosting_type': 'goss',
        'objective': 'regression',
        'num_leaves': 63,
        'min_data_in_leaf': 21,
        'max_depth': 7,
        'learning_rate': 0.01,
        'reg_alpha': 9.677537745007898,
        'feature_fraction': 0.5665320670155495,
        'bagging_fraction': 0.9855232997390695,
        'metric': 'rmse',
        'verbosity': -1,
        'top_rate': 0.9064148448434349,
        'min_child_weight': 41.9612869171337,
        'other_rate': 0.0721768246018207,
        'min_split_gain': 9.820197773625843,
        'reg_lambda': 8.2532317400459,
        'drop_seed': SEED,
    }

    return params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, required=True)
    args = parser.parse_args()

    main(args)