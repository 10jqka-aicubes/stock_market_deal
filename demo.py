# demon.py
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np

data = pd.read_csv("demo.csv", sep="\t")
df_train = data[data["rank"] >= 20000].iloc[:, 4:]
df_test = data[data["rank"] < 20000].iloc[:, 4:]

train_y = df_train["target"]
test_y = df_test["target"]
train_x = df_train.drop(["target"], axis=1)
test_x = df_test.drop(["target"], axis=1)

params = {
    "boosting_type": "gbdt",
    "metric": "rmse",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": 0,
}


model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=1000,
    learning_rate=0.1,
    num_leaves=31,
    feature_fraction=0.9,
    bagging_fraction=0.8,
    bagging_freq=5,
    verbose=-1,
    random_state=2022,
)

model.fit(
    train_x,
    train_y,
    eval_set=[(train_x, train_y), (test_x, test_y)],
    eval_metric=["rmse"],
    early_stopping_rounds=10,
    verbose=1,
)


pred = model.predict(test_x)
mse = mean_squared_error(test_y, pred)
rmse = np.sqrt(mean_squared_error(test_y, pred))
print("RMSE: ", rmse)
