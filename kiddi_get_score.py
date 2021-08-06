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
model_path = ''
feature_col = ['qLymphVal_min_12mth', 'qNeutrophilsVal_max_9mth',
       'BactInfec_avg_12mth', 'qNeutrophilsVal_max_12mth',
       'qLymphVal_min_9mth', 'zscore_age_BactInfec_sum_12mth',
       'zscore_Diabetes_BactInfec_sum_9mth', 'BactInfec_avg_9mth',
       'PDVintage', 'zscore_age_qLDLCholVal_sum_6mth',
       'zscore_age_BactInfec_sum_9mth', 'zscore_sex_BactInfec_sum_9mth',
       'BactInfec_max_9mth', 'zscore_Diabetes_qLDLCholVal_sum_6mth',
       'zscore_sex_qBicarbVal_sum_12mth',
       'zscore_Diabetes_qExch3CAPDDwellTime_sum_12mth',
       'qAlbuminVal_min_9mth', 'qAlbuminVal_min_12mth',
       'qPTHVal_sum_12mth', 'qWhBldCntVal_q1_12mth',
       'BactInfec_max_12mth', 'zscore_sex_BactInfec_sum_12mth',
       'BactInfec_med_12mth', 'qAlbuminVal_avg_9mth',
       'zscore_age_qGlucoseVal_sum_12mth',
       'zscore_Diabetes_qExch1CAPDBaxSolType_sum_3mth',
       'zscore_age_qCreatCon4HrUnit_sum_12mth', 'qLymphVal_q1_9mth',
       'zscore_age_qCreatCon4HrCorUnit_sum_12mth', 'qLymphVal_q3_12mth',
       'zscore_age_qCreatVal_sum_12mth', 'zscore_age_qWeightVal_sum_9mth',
       'zscore_sex_qCreatVal_sum_12mth',
       'zscore_age_qExch4CAPDOtherSolType_sum_12mth',
       'qWhBldCntVal_avg_12mth', 'PatientAge',
       'zscore_age_qLDLCholVal_sum_9mth',
       'zscore_sex_qLDLCholVal_sum_6mth', 'zscore_sex_qPTHVal_sum_12mth',
       'qLymphVal_q1_12mth', 'qWhBldCntVal_max_12mth',
       'qAlbuminVal_q1_12mth', 'qWhBldCntVal_max_9mth',
       'zscore_sex_qGlucoseVal_sum_12mth',
       'qExch4CAPDOtherSolType_sum_12mth', 'qAlbuminVal_q1_9mth',
       'qWhBldCntVal_q3_12mth', 'qExch3CAPDDwellTime_sum_12mth',
       'zscore_age_qPlateletsVal_sum_12mth',
       'zscore_Diabetes_BactInfec_sum_12mth',
       'zscore_Diabetes_qCreatCon4HrCorUnit_sum_12mth',
       'zscore_Diabetes_qExch4CAPDOtherSolType_sum_12mth',
       'qLDLCholVal_sum_6mth', 'qCaTotalVal_min_12mth',
       'zscore_Diabetes_qSodiumVal_sum_12mth', 'qPlateletsVal_q1_12mth',
       'qCaTotalVal_avg_12mth', 'qAlbuminVal_q3_9mth',
       'zscore_Diabetes_qALTVal_sum_12mth', 'qPulseVal_max_12mth',
       'zscore_Diabetes_qPTHVal_sum_12mth',
       'zscore_Diabetes_qPTHVal_sum_9mth', 'qCaTotalVal_min_9mth',
       'BactInfec_avg_6mth', 'qHctVal_max_9mth', 'qAlbuminVal_q3_12mth',
       'qHgbVal_avg_9mth', 'zscore_age_qWeightVolVitalVal_sum_9mth',
       'zscore_sex_qCreatCon4HrUnit_sum_12mth', 'qPTHVal_sum_9mth',
       'qCaTotalVal_max_12mth', 'zscore_sex_BactInfec_sum_6mth',
       'qCaTotalVal_q1_6mth', 'qHctVal_q1_9mth', 'qWhBldCntVal_q1_9mth',
       'qNeutrophilsVal_min_9mth',
       'zscore_sex_qWeightVolVitalVal_sum_9mth',
       'zscore_sex_qASTVal_sum_12mth',
       'zscore_sex_qExch2CAPDBaxSolType_sum_3mth',
       'zscore_Diabetes_qASTVal_sum_12mth', 'qPlateletsVal_avg_6mth',
       'qPlateletsVal_max_12mth', 'zscore_age_qNeutrophilsVal_sum_12mth',
       'zscore_sex_qTriglycVal_sum_12mth', 'qPulseVal_max_9mth',
       'qHgbVal_q3_9mth', 'qSystolic_min_12mth',
       'zscore_Diabetes_qAlkPhosphVal_sum_12mth', 'qMCVVal_q3_12mth',
       'qHgbVal_avg_12mth', 'qCreatVal_sum_12mth',
       'zscore_sex_qCreatVal_sum_9mth', 'qHctVal_max_12mth',
       'qCreatVal_max_12mth', 'zscore_sex_qExch1CAPDBaxSolType_sum_3mth',
       'qNeutrophilsVal_q1_12mth', 'zscore_sex_qLDLCholVal_sum_9mth',
       'qPhosphVal_q3_9mth', 'zscore_sex_qExch3CAPDBaxSolType_sum_3mth',
       'qLymphVal_q3_9mth']
st_data_dt = np.datetime64(date(2017, 6, 1))
end_data_dt = np.datetime64(date(2017, 6, 30))
score = kiddi.get_prob(model_path, feature_col, st_data_dt, end_data_dt)