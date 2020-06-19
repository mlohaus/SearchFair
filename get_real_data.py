#!/usr/bin/env python
import os
from random import shuffle
import pandas as pd

def get_celebA_data(load_data_size=None):
    """Load the celebA dataset.
    Source: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

    Parameters
    ----------
    load_data_size: int
        The number of points to be loaded. If None, returns all data points unshuffled.

    Returns
    ---------
    X: numpy array
        The features of the datapoints with shape=(number_points, number_features).
    y: numpy array
        The class labels of the datapoints with shape=(number_points,).
    s: numpy array
        The binary sensitive attribute of the datapoints with shape=(number_points,).
    """

    src_path = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(os.path.join(src_path, './data/celebA/list_attr_celeba.csv'), sep=';')
    df = df.rename(columns={'Male': 'sex'})

    s = df['sex']
    y = df['Smiling']
    df = df.drop(columns=['sex', 'Smiling','picture_ID'])

    X = df.to_numpy()
    y = y.to_numpy()
    s = s.to_numpy()

    if load_data_size is not None: # Don't shuffle if all data is requested
    	# shuffle the data
    	perm = list(range(0, len(y)))
    	shuffle(perm)
    	X = X[perm]
    	y = y[perm]
    	s = s[perm]

    	print("Loading only %d examples from the data" % load_data_size)
    	X = X[:load_data_size]
    	y = y[:load_data_size]
    	s = s[:load_data_size]

    X = X[:, (X != 0).any(axis=0)]

    return X, y, s

def get_adult_data(load_data_size=None):
    """Load the Adult dataset.
    Source: UCI Machine Learning Repository.

    Parameters
    ----------
    load_data_size: int
        The number of points to be loaded. If None, returns all data points unshuffled.

    Returns
    ---------
    X: numpy array
        The features of the datapoints with shape=(number_points, number_features).
    y: numpy array
        The class labels of the datapoints with shape=(number_points,).
    s: numpy array
        The binary sensitive attribute of the datapoints with shape=(number_points,).
    """

    def mapping(tuple):
    	# age, 37
    	tuple['age'] = 1 if tuple['age'] > 37 else 0
    	# workclass
    	tuple['workclass'] = 'NonPrivate' if tuple['workclass'] != 'Private' else 'Private'
    	# edunum
    	tuple['education-num'] = 1 if tuple['education-num'] > 9 else 0
    	# maritial statue
    	tuple['marital-status'] = "Marriedcivspouse" if tuple['marital-status'] == "Married-civ-spouse" else "nonMarriedcivspouse"
    	# occupation
    	tuple['occupation'] = "Craftrepair" if tuple['occupation'] == "Craft-repair" else "NonCraftrepair"
    	# relationship
    	tuple['relationship'] = "NotInFamily" if tuple['relationship'] == "Not-in-family" else "InFamily"
    	# race
    	tuple['race'] = 'NonWhite' if tuple['race'] != "White" else 'White'
    	# hours per week
    	tuple['hours-per-week'] = 1 if tuple['hours-per-week'] > 40 else 0
    	# native country
    	tuple['native-country'] = "US" if tuple['native-country'] == "United-States" else "NonUS"
    	return tuple


    src_path = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(os.path.join(src_path, './data/adult/adult.csv'))
    df = df.drop(['fnlwgt', 'education', 'capital-gain', 'capital-loss'], axis=1)
    df = df.apply(mapping, axis=1)

    sensitive_attr_map = {'Male': 1, 'Female': -1}
    label_map = {'>50K': 1, '<=50K': -1}

    x_vars = ['age','workclass','education-num','marital-status','occupation','relationship','race','hours-per-week','native-country']

    s = df['sex'].map(sensitive_attr_map).astype(int)
    y = df['income'].map(label_map).astype(int)


    x = pd.DataFrame(data=None)
    for x_var in x_vars:
    	x = pd.concat([x, pd.get_dummies(df[x_var],prefix=x_var, drop_first=False)], axis=1)

    X = x.to_numpy()
    s = s.to_numpy()
    y = y.to_numpy()

    if load_data_size is not None: # Don't shuffle if all data is requested
        # shuffle the data
        perm = list(range(0, len(y)))
        shuffle(perm)
        X = X[perm]
        y = y[perm]
        s = s[perm]

        print("Loading only %d examples from the data" % load_data_size)
        X = X[:load_data_size]
        y = y[:load_data_size]
        s = s[:load_data_size]

    X = X[:, (X != 0).any(axis=0)]

    return X, y, s


def get_communities_crime_data(load_data_size=None):
    """Load the Communities and Crime dataset.
    Source: UCI Machine Learning Repository

    Parameters
    ----------
    load_data_size: int
        The number of points to be loaded. If None, returns all data points unshuffled.

    Returns
    ---------
    X: numpy array
        The features of the datapoints with shape=(number_points, number_features).
    y: numpy array
        The class labels of the datapoints with shape=(number_points,).
    s: numpy array
        The binary sensitive attribute of the datapoints with shape=(number_points,).
    """

    src_path = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(os.path.join(src_path, './data/crimeCommunities/communities_data.csv'))

    df['ViolentCrimesPerPop'] = df['ViolentCrimesPerPop'].apply(lambda x: -1 if x <= 0.24 else 1)
    df['racePctWhite'] = df['racePctWhite'].apply(lambda x: 'other' if x <= 0.75 else 'white')
    df = df.drop(columns=['state','county','community', 'communityname string', 'fold', 'OtherPerCap',#'medIncome', 'pctWWage', 'pctWInvInc','medFamInc',
    					  'LemasSwornFT','LemasSwFTPerPop','LemasSwFTFieldOps','LemasSwFTFieldPerPop','LemasTotalReq',
    					  'LemasTotReqPerPop','PolicReqPerOffic','PolicPerPop','RacialMatchCommPol',
    					  'PctPolicWhite','PctPolicBlack','PctPolicHisp','PctPolicAsian','PctPolicMinor','OfficAssgnDrugUnits',
    					  'NumKindsDrugsSeiz','PolicAveOTWorked','PolicCars','PolicOperBudg','LemasPctPolicOnPatr','LemasGangUnitDeploy','LemasPctOfficDrugUn','PolicBudgPerPop'])
    #29 attributes are dropped because of missing values in these features, or because they contain IDs or names

    df = df.rename(columns={'racePctWhite': 'race'})

    sensitive_attr_map = {'white': 1, 'other': -1}

    s = df['race'].map(sensitive_attr_map).astype(int)
    y = df['ViolentCrimesPerPop']

    df = df.drop(columns=['race', 'ViolentCrimesPerPop'])

    x = pd.DataFrame(data=None)
    for name in df.columns:
    	x = pd.concat([x, normalize(x=df[name])], axis=1)

    X = x.to_numpy()
    y = y.to_numpy()
    s = s.to_numpy()

    if load_data_size is not None: # Don't shuffle if all data is requested
        # shuffle the data
        perm = list(range(0, len(y)))
        shuffle(perm)
        X = X[perm]
        y = y[perm]
        s = s[perm]

        print("Loading only %d examples from the data" % load_data_size)
        X = X[:load_data_size]
        y = y[:load_data_size]
        s = s[:load_data_size]

    X = X[:, (X != 0).any(axis=0)]

    return X, y, s

def get_german_data(load_data_size=None):
    """Load the German Credit dataset.
    Source: UCI Machine Learning Repository.

    Parameters
    ----------
    load_data_size: int
        The number of points to be loaded. If None, returns all data points unshuffled.

    Returns
    ---------
    X: numpy array
        The features of the datapoints with shape=(number_points, number_features).
    y: numpy array
        The class labels of the datapoints with shape=(number_points,).
    s: numpy array
        The binary sensitive attribute of the datapoints with shape=(number_points,).
    """

    src_path = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(os.path.join (src_path, './data/german/german.csv'))

    sexdict = {'A91': 'male', 'A93': 'male', 'A94': 'male',
    		   'A92': 'female', 'A95': 'female'}
    df = df.assign(personal_status=df['personal_status'].replace(to_replace=sexdict))
    df = df.rename(columns={'personal_status': 'sex'})

    sensitive_attr_map = {'male': 1, 'female': -1}
    label_map = {1: 1, 2: -1}

    s = df['sex'].map(sensitive_attr_map).astype(int)
    y = df['credit'].map(label_map).astype(int)

    x_vars_categorical = [
    	'status','credit_history','purpose','savings','employment',
        'other_debtors','property','installment_plans','housing','skill_level',
        'telephone','foreign_worker'
    ]

    x_vars_ordinal = [
        'month','credit_amount','investment_as_income_percentage',
        'residence_since','age','number_of_credits','people_liable_for'
    ]

    x = pd.DataFrame(data=None)
    for x_var in x_vars_ordinal:
    	x = pd.concat([x, normalize(x=df[x_var])], axis=1)
    for x_var in x_vars_categorical:
    	x = pd.concat([x, pd.get_dummies(df[x_var],prefix=x_var, drop_first=False)], axis=1)

    X = x.to_numpy()
    y = y.to_numpy()
    s = s.to_numpy()

    if load_data_size is not None: # Don't shuffle if all data is requested
        # shuffle the data
        perm = list(range(0, len(y)))
        shuffle(perm)
        X = X[perm]
        y = y[perm]
        s = s[perm]

        print("Loading only %d examples from the data" % load_data_size)
        X = X[:load_data_size]
        y = y[:load_data_size]
        s = s[:load_data_size]

    X = X[:, (X != 0).any(axis=0)]

    return X, y, s

def get_compas_data(load_data_size=None):
    """Load the Compas dataset.
    Source: Propublica Github repository: https://github.com/propublica/compas-analysis

    Parameters
    ----------
    load_data_size: int
        The number of points to be loaded. If None, returns all data points unshuffled.

    Returns
    ---------
    X: numpy array
        The features of the datapoints with shape=(number_points, number_features).
    y: numpy array
        The class labels of the datapoints with shape=(number_points,).
    s: numpy array
        The binary sensitive attribute of the datapoints with shape=(number_points,).
    """

    src_path = os.path.dirname(os.path.realpath (__file__))
    df = pd.read_csv (os.path.join(src_path, './data/compas/propublica-recidivism.csv'))

    """Preprocessing according to https://github.com/propublica/compas-analysis"""
    df = df[(df.days_b_screening_arrest <= 30) &
     					  (df.days_b_screening_arrest >= -30) &
     					  (df.is_recid != -1) &
     					  (df.c_charge_degree != '0') &
     					  (df.score_text != 'N/A')]

    df = df.drop(columns=['is_recid', 'decile_score', 'score_text'])

    racedict = {'Caucasian': 'White', 'Other': 'NonWhite', 'African-American': 'NonWhite',
    		   'Hispanic': 'NonWhite', 'Asian': 'NonWhite', 'Native American': 'NonWhite'}
    df = df.assign(race=df['race'].replace(to_replace=racedict))

    sensitive_attr_map = {'White': 1, 'NonWhite': -1}
    label_map = {1: 1, 0: -1}

    s = df['race'].map(sensitive_attr_map).astype(int)
    y = df['two_year_recid'].map(label_map).astype(int)

    """Features chosen like Zafar et al. (2017)"""
    x_vars_categorical = ['age_cat', 'c_charge_degree', 'sex']

    x_vars_ordinal = ['age', "priors_count"]

    x = pd.DataFrame(data=None)
    for x_var in x_vars_ordinal:
    	x = pd.concat([x, normalize(df[x_var])], axis=1)
    for x_var in x_vars_categorical:
    	x = pd.concat([x, pd.get_dummies(df[x_var],prefix=x_var, drop_first=False)], axis=1)

    X = x.to_numpy()
    y = y.to_numpy()
    s = s.to_numpy()

    if load_data_size is not None: # Don't shuffle if all data is requested
    	# shuffle the data
    	perm = list(range(0, len(y)))
    	shuffle(perm)
    	X = X[perm]
    	y = y[perm]
    	s = s[perm]

    	print("Loading only %d examples from the data" % load_data_size)
    	X = X[:load_data_size]
    	y = y[:load_data_size]
    	s = s[:load_data_size]

    X = X[:, (X != 0).any(axis=0)]

    return X, y, s

def get_dutch_data(load_data_size=None):
    """Load the Dutch Census dataset.
    Source: https://web.archive.org/web/20180108214635/https://sites.google.com/site/conditionaldiscrimination/

    Parameters
    ----------
    load_data_size: int
        The number of points to be loaded. If None, returns all data points unshuffled.

    Returns
    ---------
    X: numpy array
        The features of the datapoints with shape=(number_points, number_features).
    y: numpy array
        The class labels of the datapoints with shape=(number_points,).
    s: numpy array
        The binary sensitive attribute of the datapoints with shape=(number_points,).
    """

    src_path = os.path.dirname (os.path.realpath (__file__))
    df = pd.read_csv (os.path.join (src_path, './data/dutch/dutch.csv'))

    sensitive_attr_map = {2: 1, 1: -1}
    label_map = {'5_4_9': 1, '2_1': -1}

    s = df['sex'].map(sensitive_attr_map).astype(int)
    y = df['occupation'].map(label_map).astype(int)

    x_vars_categorical = [
    	'household_position',
    	'household_size',
    	'citizenship',
    	'country_birth',
    	'economic_status',
    	'cur_eco_activity',
    	'Marital_status'
    ]

    x_vars_ordinal = [
    	'age',
    	'prev_residence_place',
    	'edu_level'
    ]

    x = pd.DataFrame (data=None)
    for x_var in x_vars_ordinal:
    	x = pd.concat ([x, normalize(x=df[x_var])], axis=1)
    for x_var in x_vars_categorical:
    	x = pd.concat([x, pd.get_dummies(df[x_var],prefix=x_var, drop_first=False)], axis=1)

    X = x.to_numpy()
    y = y.to_numpy()
    s = s.to_numpy()

    if load_data_size is not None: # Don't shuffle if all data is requested
    	# shuffle the data
    	perm = list(range(0, len(y)))
    	shuffle(perm)
    	X = X[perm]
    	y = y[perm]
    	s = s[perm]

    	print("Loading only %d examples from the data" % load_data_size)
    	X = X[:load_data_size]
    	y = y[:load_data_size]
    	s = s[:load_data_size]

    X = X[:, (X != 0).any(axis=0)]

    return X, y, s

def normalize(x):
	# scale to [-1, 1]
	x_ = (x - x.min()) / (x.max() - x.min()) * 2 - 1
	return x_

if __name__=='__main__':

    x,y,s = get_celebA_data()

    print(x.shape)
    print(s)
    print(y)
