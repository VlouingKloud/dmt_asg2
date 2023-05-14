import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression

def _get_weekday(x):
    date_obj = datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    days_of_week = {
            'Monday': 1,
            'Tuesday': 2,
            'Wednesday': 3,
            'Thursday': 4,
            'Friday': 5,
            'Saturday': 6,
            'Sunday': 7
        }

    return days_of_week[date_obj.strftime("%A")]

def _get_hour(x):
    date_obj = datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    return int(date_obj.strftime("%H"))

def _is_weekdend(x):
    return x == 'Saturday' or x == 'Sunday'

def add_weekday(df):
    df['weekday'] = df['date_time'].apply(lambda x: _get_weekday(x))
    return df

def add_isweekend(df):
    df['is_weekend'] = df['weekday'].apply(lambda x: _is_weekdend(x))
    return df

def add_hour(df):
    df['hour'] = df['date_time'].apply(lambda x: _get_hour(x))
    return df

def drop_columns_with_many_nans(df, th = 0.5):
    todel = []
    for i in df.columns:
        if sum(df[i].isna())/len(df[i]) >= th:
            todel.append(i)
    return df.drop(todel, axis = 1)

def _fillnasingle(df, col, val):
    df[col] = df[col].fillna(val)
    return df

def fill_prop_location_score2_by_lr(df, model = None):
    df_train = df[df["prop_location_score2"].notnull()]
    X_train = df_train[["prop_location_score1"]]
    y_train = df_train["prop_location_score2"]
    if not model:
        model = LinearRegression()
        model.fit(X_train, y_train)
    df_test = df[df["prop_location_score2"].isnull()]
    X_test = df_test[["prop_location_score1"]]
    y_pred = model.predict(X_test)
    df.loc[df["prop_location_score2"].isnull(), "prop_location_score2"] = y_pred
    return df, model

def main():
    ### process training data
    df = pd.read_csv("./training_set_VU_DM.csv")

    df = drop_columns_with_many_nans(df)
    df = df.drop(['position'], axis = 1)

    df = add_weekday(df)
    df = add_isweekend(df)
    df = add_hour(df)

    df = df.drop(['date_time'], axis = 1)

    prop_review_score_mean = df['prop_review_score'].mean()
    df = _fillnasingle(df, "prop_review_score", prop_review_score_mean)

    orig_destination_distance_median = df['orig_destination_distance'].median()
    df = _fillnasingle(df, "orig_destination_distance", orig_destination_distance_median)

    df, mdl = fill_prop_location_score2_by_lr(df)

    df.to_csv("train_stage1.csv", index=False)

    ### process testing data
    df = pd.read_csv("./test_set_VU_DM.csv")

    df = drop_columns_with_many_nans(df)

    df = add_weekday(df)
    df = add_isweekend(df)
    df = add_hour(df)

    df = _fillnasingle(df, "prop_review_score", prop_review_score_mean)

    df = _fillnasingle(df, "orig_destination_distance", orig_destination_distance_median)

    df, mdl = fill_prop_location_score2_by_lr(df, mdl)

    df.to_csv("test_stage1.csv", index=False)


main()
