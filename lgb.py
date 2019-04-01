import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

from constants import SEED, DEVICE


def get_params(varparams):
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'bagging_seed': SEED,
        'feature_fraction_seed': SEED,
        'seed': SEED,
        'verbosity': -1,
        'device': DEVICE,
    }
    params.update(varparams)

    params_in_train = {
        'num_boost_round': 10000,
        'early_stopping_rounds': 100,
        'verbose_eval': 100,
    }

    n_splits = 5

    return params, params_in_train, n_splits


def run_lgb(train_X, train_y, test_X, params):
    params, params_in_train, n_splits = get_params(params)

    print('[START] start training!')
    print('[PARAM] params: {}'.format(params))

    kf = KFold(n_splits=n_splits, random_state=SEED, shuffle=True)

    oof = np.zeros(len(train_X))
    predictions = np.zeros(len(test_X))
    scores = {'train': [], 'valid': []}
    features = list(train_X.columns)
    feature_importance_df = pd.DataFrame(columns=['fold', 'feature', 'importance'])

    for fold, (dev_idx, val_idx) in enumerate(kf.split(train_X)):
        print('fold: {}/{}'.format(fold+1, kf.n_splits))

        dev_X, val_X = train_X.loc[dev_idx,:], train_X.loc[val_idx,:]
        dev_y, val_y = train_y[dev_idx], train_y[val_idx]

        lgdev = lgb.Dataset(dev_X, dev_y)
        lgval = lgb.Dataset(val_X, val_y)

        model = lgb.train(params, lgdev, valid_sets=[lgdev, lgval], **params_in_train)
        scores['train'].append(model.best_score['training']['rmse'])
        scores['valid'].append(model.best_score['valid_1']['rmse'])
        oof[val_idx] = model.predict(val_X, num_iteration=model.best_iteration)

        fold_feature_importance_df = pd.DataFrame(columns=['fold', 'feature', 'importance'])
        fold_feature_importance_df['feature'] = features
        fold_feature_importance_df['importance'] = model.feature_importance()
        fold_feature_importance_df['fold'] = fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_feature_importance_df], axis=0)
        
        predictions += model.predict(test_X, num_iteration=model.best_iteration) / kf.n_splits
    
    cv_score = mean_squared_error(oof, train_y)**0.5

    print('##### RESULTS #####')
    print('Num features: {}'.format(len(features)))
    print('Num folds: {}'.format(n_splits))
    print('Train Scores: mean {:.5f}, max {:.5f}, min {:.5f}, std {:.5f}'.format(
        np.mean(scores['train']), np.max(scores['train']), np.min(scores['train']), np.std(scores['train'])
    ))
    print('Valid Scores: mean {:.5f}, max {:.5f}, min {:.5f}, std {:.5f}'.format(
        np.mean(scores['valid']), np.max(scores['valid']), np.min(scores['valid']), np.std(scores['valid'])
    ))
    print('CV Score: {:<8.5f}'.format(cv_score))

    return oof, predictions, feature_importance_df
