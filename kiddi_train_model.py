from Kiddi_ml import Kiddi
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd
import datetime
from dateutil.relativedelta import relativedelta
import calendar
from datetime import date
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, roc_curve, auc, precision_recall_curve
from sklearn import metrics
from scipy import stats
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype
from scipy.stats import zscore
import pickle
IS_RAW_PATH = '/Users/metis_sotangkur/Desktop/Kiddi_ds/Data/is_raw_idd.xlsx'
IWP_PATH = '/Users/metis_sotangkur/Desktop/Kiddi_ds/Data/iwp_raw_idd.xlsx'
IWES_PATH = '/Users/metis_sotangkur/Desktop/Kiddi_ds/Data/iwes_raw_idd.xlsx'
CENSUS_PATH = '/Users/metis_sotangkur/Desktop/Kiddi_ds/Data/census_idd.xlsx'
kiddi = Kiddi(IS_RAW_PATH, IWP_PATH, IWES_PATH, CENSUS_PATH)
kiddi.clean_data()
kiddi.gen_population()
kiddi.gen_target()
kiddi.partition_is_raw()
kiddi.feature_engineer_ts()
kiddi.feature_engineer_segment()

kiddi.join_target()
st_train_dt = np.datetime64(date(2016, 2, 1) )
end_train_dt = np.datetime64(date(2017, 4, 30))
st_val_dt = np.datetime64(date(2016, 2, 1))
end_val_dt = np.datetime64(date(2017, 4, 30))
st_test_dt = np.datetime64(date(2017, 5, 1))
end_test_dt = np.datetime64(date(2017, 6, 30))
kiddi.prep_data_fn(st_train_dt, end_train_dt, st_val_dt, end_val_dt, st_test_dt, end_test_dt)

all_col = kiddi.get_all_columns()
print(len(all_col))
param = dict()
param['tree_num'] = 100
param['depth'] = 10
model, model_path = kiddi.train_model('random_forest', param, all_col)

ft_imp = model.feature_importances_
new_col = []
for c, f in zip(ft_imp, all_col):
    new_col.append((c, f))
new_col.sort(reverse=True)
top_100_col = np.array(new_col[:100])[:, 1]
param = dict()
param['tree_num'] = 100
param['depth'] = 10
model_top100_list, model_path_100 = kiddi.train_model('random_forest', param, top_100_col)

ft_imp = model_top100_list.feature_importances_
new_col = []
for c, f in zip(ft_imp, top_100_col):
    new_col.append((c, f))
new_col.sort(reverse=True)
new_col = np.array(new_col)
ft_tmp = pd.DataFrame(new_col, columns=['value', 'feature'])
ft_tmp = ft_tmp[['feature', 'value']]
ft_tmp['cum_value'] = ft_tmp['value'].cumsum()

print(ft_tmp)