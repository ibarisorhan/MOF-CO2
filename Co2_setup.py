#Standard Libraries
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
import statistics as stat
import numpy as np
import seaborn as sns
import random 

#Dataset Trimming and Transformation
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler

#Regressors
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.neural_network import MLPRegressor

#Evaluation
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#Additions for Figures
from matplotlib.patches import Rectangle

groups = {'MOF & Target': ['MOF', 'Uptake', 'Pressure'],
 'A': ['H', 'C', 'N', 'F', 'Cl', 'Br', 'V', 'Cu', 'Zn', 'Zr'],
 'B': ['ASA_m^2/g_15prb',
  'NASA_m^2/g_15prb',
  'AV_cm^3/g_15prb',
  'NAV_cm^3/g_15prb',
  'POAV_cm^3/g_15prb',
  'PONAV_cm^3/g_15prb',
  'PLD',
  'LCD',
  'LFPD',
  'Density',
  'Volume'],
 'C': ['TotalDegreeOfUnsaturation',
  'MetallicPercentage',
  'OxygenToMetalRatio',
  'ElectronegativeToTotalRatio',
  'WeightedElectronegativityPerAtom',
  'NitrogenToOxygen'],
 'D': ['Epoch40',
  'Epoch1000',
  'Epoch4000',
  'Epoch40Ave',
  'Epoch4000Ave',
  'Epoch1000Ave'],
 'E': ['HenryCoeff_CO2']}


descriptors_all_rename_concise = {
 'H':'H',
 'C':'C',
 'N':'N',
 'F':'F',
 'Cl':'Cl',
 'Br':'Br',
 'V':'V',
 'Cu':'Cu',
 'Zn':'Zn',
 'Zr':'Zr',
 'ASA_m^2/g_15prb':'ASA ($m^2$/g)',
 'NASA_m^2/g_15prb':'NASA ($m^2$/g)',
 'AV_cm^3/g_15prb':'AV ($cm^3$/g)',
 'NAV_cm^3/g_15prb':'NAV ($cm^3$/g)',
 'POAV_cm^3/g_15prb':'POAV ($cm^3$/g)',
 'PONAV_cm^3/g_15prb':'PONAV ($cm^3$/g)',
 'PLD':'PLD',
 'LCD':'LCD',
 'LFPD':'LFPD',
 'Density':'Density',
 'Volume':'Volume',
 'TotalDegreeOfUnsaturation':'Degree of Unsaturation',
 'MetallicPercentage' : 'Metal (%)',
 'OxygenToMetalRatio' : 'Oxygen:Metal Ratio',
 'ElectronegativeToTotalRatio' : 'Electronegative Ratio',
 'WeightedElectronegativityPerAtom' : 'Weighted Electronegativity',
 'NitrogenToOxygen' : 'Nitrogen:Oxygen Ratio',
 'Epoch40':'$EPoCh_{40}$',
 'Epoch1000':'$EPoCh_{1000}$',
 'Epoch4000':'$EPoCh_{4000}$',
 'Epoch40Ave':'$EPoCh_{40,Average}$',
 'Epoch4000Ave':'$EPoCh_{4000,Average}$',
 'Epoch1000Ave':'$EPoCh_{1000,Average}$',
 'HenryCoeff_CO2': '$K_{H,CO_2}$'
}


descriptors_all_rename = {
 'H':'H',
 'C':'C',
 'N':'N',
 'F':'F',
 'Cl':'Cl',
 'Br':'Br',
 'V':'V',
 'Cu':'Cu',
 'Zn':'Zn',
 'Zr':'Zr',
 'ASA_m^2/g_15prb':'ASA ($m^2$/g)',
 'NASA_m^2/g_15prb':'NASA ($m^2$/g)',
 'AV_cm^3/g_15prb':'AV ($cm^3$/g)',
 'NAV_cm^3/g_15prb':'NAV ($cm^3$/g)',
 'POAV_cm^3/g_15prb':'POAV ($cm^3$/g)',
 'PONAV_cm^3/g_15prb':'PONAV ($cm^3$/g)',
 'ASA_m^2/g':'ASA ($m^2$/g)',
 'NASA_m^2/g':'NASA ($m^2$/g)',
 'AV_cm^3/g':'AV ($cm^3$/g)',
 'NAV_cm^3/g':'NAV ($cm^3$/g)',
 'POAV_cm^3/g':'POAV ($cm^3$/g)',
 'PONAV_cm^3/g':'PONAV ($cm^3$/g)',
 'PLD':'PLD',
 'LCD':'LCD',
 'LFPD':'LFPD',
 'Density':'Density',
 'Volume':'Volume',
 'TotalDegreeOfUnsaturation':'Degree of Unsaturation',
 'MetallicPercentage' : 'Metal (%)',
 'OxygenToMetalRatio' : 'Oxygen:Metal Ratio',
 'ElectronegativeToTotalRatio' : 'Electronegative Ratio',
 'WeightedElectronegativityPerAtom' : 'Weighted Electronegativity',
 'NitrogenToOxygen' : 'Nitrogen:Oxygen Ratio',
 'Epoch40':'$EPoCh_{40}$',
 'Epoch1000':'$EPoCh_{1000}$',
 'Epoch4000':'$EPoCh_{4000}$',
 'Epoch40Ave':'$EPoCh_{40,Average}$',
 'Epoch4000Ave':'$EPoCh_{4000,Average}$',
 'Epoch1000Ave':'$EPoCh_{1000,Average}$',
 'HenryCoeff_CO2': '$K_{H,CO_2}$'
}


baseline_descriptors = groups['A'] +  groups['B'] + groups['C']
descriptors_all = groups['A'] +  groups['B'] + groups['C']+ groups['D']+ groups['E']
all_cols = groups['MOF & Target'] + descriptors_all
energy_descriptors = groups['E']
non_energetic_descriptors = groups['A'] +  groups['B'] + groups['C']+ groups['D']




def evaluate_ML(ml,X,y,axes,title="",threshold = 1,add_rects =True): 
    GREEN = "palegreen" #"#c4ff52"
    RED = "salmon" #"#ff526e"
    BGA = 0.5 #Background Alpha
    A = 0.4
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = SEED) 
    ml.fit(X_train,y_train)        
    train_pred = ml.predict(X_train)
    train_r2   = r2_score(y_train,train_pred)
    test_pred  = ml.predict(X_test)
    test_r2    = r2_score(y_test,test_pred)

    axes.plot(y_train,train_pred,'o',label = f'Train $R^2$={train_r2:4.3}', alpha = A)
    axes.plot(y_test,test_pred,'o',label = f'Test $R^2$={test_r2:4.3}', alpha = A)
    axes.legend(loc='lower right')

    limit = max([max(y_train),max(train_pred),max(y_test),max(test_pred)])
    axes.set_xlim((0,limit))
    axes.set_ylim((0,limit))
    axes.plot((0,limit),(0,limit),'--k')
    axes.set_title(title,loc='left')
    if add_rects:
        rect1 = Rectangle((0,0), threshold, threshold, color='none', fc = f'{GREEN}',lw = 2, alpha = BGA)
        rect2 = Rectangle((threshold,threshold), limit-threshold, limit-threshold, color='none', fc = f'{GREEN}',lw = 2, alpha = BGA)
        rect3 = Rectangle((0,threshold), threshold, limit-threshold, color='none', fc = f'{RED}',lw = 2, alpha = BGA)
        rect4 = Rectangle((threshold,0), limit-threshold, threshold, color='none', fc = f'{RED}',lw = 2, alpha = BGA)
        axes.add_patch(rect1)
        axes.add_patch(rect2)
        axes.add_patch(rect3)
        axes.add_patch(rect4)
        
    axes.set_xlabel("Simulated (mmol/g)")
    axes.set_ylabel("Predicted (mmol/g)")
    
    pseudo_class = {"TN":0,'TP':0,'FN':0,'FP':0}
    for i in range(len(y_test)):
        tru = list(y_test)[i]
        prd = list(test_pred)[i]
        if tru < threshold:
            if prd < threshold:
                pseudo_class['TN'] +=1
            else:
                pseudo_class['FP'] += 1
        else:
            if prd < threshold:
                pseudo_class['FN'] +=1
            else:
                pseudo_class['TP'] += 1
    return pseudo_class



def X_y_from_pressure(df,pressure,descriptors=descriptors_all,target='Uptake'):
    current = df[df.Pressure == pressure] 
    X = current[descriptors]
    y = current[target]
    return X, y


def pcm(pseudo_class): #pseudo classification metrics
    precision = pseudo_class['TP'] / (pseudo_class['TP'] + pseudo_class['FP'])
    recall = pseudo_class['TP'] / (pseudo_class['TP'] + pseudo_class['FN'])
    print(f"Recall   : {recall:4.3f}")
    print(f"Precision: {precision:4.3f}")


SEED = 2559288933 #obtained from random.randint(0,2**32 - 1); used for train-test splitting
SEED2= 3113586555 #obtained same way as above, used for random states of algorithms 

Models = {
    'AdaBoostRegressor': AdaBoostRegressor(random_state = SEED2),
    'BaggingRegressor': BaggingRegressor(random_state = SEED2),
    'ExtraTreesRegressor': ExtraTreesRegressor(random_state = SEED2),
    'GradientBoostingRegressor': GradientBoostingRegressor(random_state = SEED2),
    'HistGradientBoostingRegressor': HistGradientBoostingRegressor(random_state = SEED2),
    'RandomForestRegressor': RandomForestRegressor(random_state = SEED2),
    'LinearRegression': LinearRegression(),
    'BayesianRidge': BayesianRidge()
}