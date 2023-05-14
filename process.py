import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    ### prepare training data
    df = pd.read_csv('./train_stage1.csv')
    weight_click = 1
    weight_booking = 5
    df['target'] = (weight_click * df['click_bool']) + (weight_booking * df['booking_bool'])
    df = df.drop(['click_bool', 'booking_bool'], axis = 1)

    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

    features = set(df.columns)
    features.remove("target")
    features = list(features)
    target = 'target'

    lgb_train = lgb.Dataset(train_data[features], train_data[target], group=train_data.groupby('srch_id').size())
    lgb_test = lgb.Dataset(test_data[features], test_data[target], group=test_data.groupby('srch_id').size())

    ### build model
    params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': 10,
            'max_position': 20,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'verbose': 1
            }
    model = lgb.train(params, lgb_train, num_boost_round=1000, valid_sets=[lgb_train, lgb_test], early_stopping_rounds=50)

    ### test, note that there is no validation yet
    del df
    del train_data
    del test_data
    df = pd.read_csv('./test_stage1.csv')
    df['predicted_booking_prob'] = model.predict(df[features])
    df = df.sort_values(['srch_id', 'predicted_booking_prob'], ascending=[True, False])
    df = df[['srch_id', 'prop_id']]
    df.to_csv("result.csv", index=False)


main()
