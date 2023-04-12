'''
File to connect to SQL data based and read relvant table
Author: Amir Yazdavar
'''


"""
For LR models:
    (1) Comment out (#) Model fit and plots (lines 223 > 254); uncomment regularization penalty space (lines 206 > 221)
        (a) run model
    (2) Comment regularization penalty space (lines 206 > 221); Uncomment out Model fit and plots (lines 223 > 254);
        (a) run model

For XGB models:
    (1) run model 1 and adjust the feature importance threshold (line 290) to remove low/negative features;
        (a) rerun the model
        (b) tweak hyperparameters to improve performance (i.e. n_estimators, learning_rate, max_depth); rerun until
        best performance achieved
    (2) run model 2 (which uses the features selected from the first model), adjust feature importance
        (a) print/copy xgb2_feats and paste them into unwanted = {}
        (b) rerun model 2; tweak feature importance threshold and hyperparameters as necessary

For RF models:
    (1) model 1: Comment out (#) the GridSearch and Model fit (lines 417 > 444); Run the model
        (a) model 1: Comment out Randomized Search (lines 384 > 415), uncomment Grid Search (lines 417 > 433)
        (b) model 1: adjust Grid Search around the results printed: "RF Best Params", generally using +/- on either side
            (i.) Best Params: {'n_estimators': 600, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features':
            'sqrt', 'max_depth': 30, 'class_weight': 'balanced', 'bootstrap': False}
            (ii.) Example: param_grid = {'bootstrap': [False], 'max_depth': [20,30,40], 'max_features': ['sqrt'],
            'min_samples_leaf': [1,2], 'min_samples_split': [4,5,6], 'class_weight':['balanced'], 'n_estimators': [
            400,600,800]
            (iii.) Rerun model 1
        (c) model 1: Comment out Grid Search (lines 417 > 433); Uncomment Model Fit (lines 435 > 444)
            (i.) Replace RandomForestClassifier parameters (line 435) with Best Params from grid search
            (ii.) run model;
            (iii.) adjust feature importance thresholds (line 440); re-run as necessary
    (2) model 2: follow steps for model 1 to run Randomized Search, Grid Search, and Feature Selection
            (i.) comment out the 'unwanted' and ft variable lines for the first run

"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
from rfpimp import *
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

plt.rc("font", size=14)
import seaborn as sns
import sys
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

# ignore Deprecation Warning
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

PATH = '/home/ubuntu/amir/depression_sdh/depression_sdh/Joe_results/'
#PATH = '/XXXXX/venv/'  # Root path
# PATH2 = '/XXXXX/XXXXX/Documents/'  # Backup img path
PATH2 = '/home/ubuntu/amir/depression_sdh_model_results'

cdrn = pd.read_csv(PATH + 'cdrn_ccs_modified_label2.csv', encoding='utf-8')  # Read Data into Python

""" 
=== MODEL PARAMETERS ===
full_model:
    True: all data points; [1:X] cases:controls
    False: restrictions applied to data (age, visit_count, date_diff, diagnoses); [1:X] cases:controls

primary_dx:
    True: primary conditions only
    False: primary & secondary conditions

subsample ([1:1] cases:controls):
    True: subsample
    False: full data set [1:X]

smote ([1:1]) cases:controls:
    True: synthetically up-sample cases, down-sample controls
    False: full data set [1:X]
"""
# Apply Model Parameters
full_model = False
primary_dx = False
subsample = False
smote = False
SEED = 42

if subsample is True and smote is True:
    raise NameError('SUBSAMPLE and SMOTE cannot -both- be True [line 89, 90]')

# Establish Save File Suffix
if full_model is True:
    model_type = 'full'
else:
    model_type = 'restrict'

if primary_dx is True:
    dx_type = 'primary'
else:
    dx_type = 'primsec'

if subsample is True:
    sub = 'sub'
elif smote is True:
    sub = 'smote'
else:
    sub = 'all'

outfile = model_type + '_' + dx_type + '_' + sub + '.png'


def data_processing(df):
    drugs = pd.read_csv(PATH + 'psych_outpt_drug_exp_1218.csv', encoding='utf-8', index_col=False)
    drugs = drugs.drop(drugs.columns[0], axis=1)
    procs = pd.read_csv(PATH + 'cdrn_procs_grouped_1218.csv', encoding='utf-8', index_col=False)

    features = ['person_id', 'tuberculosis', 'septic', 'bact_infec', 'mycoses', 'hiv', 'hepatitis', 'viral_infec',
                'other_infec', 'sti', 'screen_infec', 'head_ca', 'esophagus_ca', 'stomach_ca', 'colon_ca', 'rectum_ca',
                'liver_ca', 'pancreas_ca', 'gi_ca', 'lung_ca', 'resp_ca', 'bone_ca', 'melanoma', 'nonepi_skin_ca',
                'breast_ca', 'uterus_ca', 'cervix_ca', 'ovary_ca', 'fem_genital_ca', 'prostate_ca', 'testes_ca',
                'male_genital_ca', 'bladder_ca', 'kidney_ca', 'urinary_ca', 'brain_ca', 'thyroid_ca', 'hodgkins_lymph',
                'non_hodgkins_lymph', 'leukemia', 'mult_myeloma', 'other_ca', 'secndry_malig', 'malig_neoplasm',
                'neoplasm_unspec', 'maint_chemo', 'ben_neoplasm_uterus', 'other_ben_neoplasm', 'thyroid', 'dm_wo_comp',
                'dm_w_comp', 'other_endocrine', 'nutrition', 'lipid_metabo', 'gout', 'fluid_electrolyte',
                'cyst_fibrosis', 'immunity', 'other_metabo', 'other_anemia', 'post_hemorr_anemia', 'sickle_cell',
                'coag_anemia', 'wbc_disease', 'other_heme', 'meningitis_no  tb', 'encephalitis_notb', 'other_cns',
                'parkinsons', 'mult_scler', 'other_hered_degen', 'paralysis', 'epilepsy', 'headache', 'coma',
                'cataract', 'retinopathy', 'glaucoma', 'blindness', 'eye_inflam', 'other_eye', 'otitis_media', 'dizzy',
                'other_ear_sense', 'other_ns_disorder', 'heart_valve', 'peri_endo_carditis', 'essential_htn',
                'htn_w_comp', 'acute_mi', 'coronary_athero', 'chest_pain_nos', 'pulmonary_hd', 'other_heart_disease',
                'conduction', 'cardiac_dysrhythm', 'cardiac_arrest', 'chf', 'acute_cvd', 'occlu_cereb_artery',
                'other_cvd', 'tran_cereb_isch', 'late_effect_cvd', 'pvd', 'artery_aneurysm', 'artery_embolism',
                'other_circ', 'phlebitis', 'varicose_vein', 'hemorrhoid', 'other_vein_lymph', 'pneumonia', 'influenza',
                'acute_tonsil', 'acute_bronch', 'upper_resp_infec', 'copd', 'asthma', 'asp_pneumonitis', 'pneumothorax',
                'resp_failure', 'lung_disease', 'other_low_resp', 'other_up_resp', 'intestinal_infec', 'teeth_jaw',
                'mouth_disease', 'esophagus', 'gastro_ulcer', 'gastritis', 'other_stomach_duo', 'appendicitis',
                'hernia_abd', 'regional_enteriritis', 'intestinal_obstruct', 'diverticulitis', 'anal_condition',
                'peritonitis', 'biliary_tract', 'other_liver', 'pancreatic', 'gastro_hemorrhage', 'noninfec_gastro',
                'other_gastro', 'nephritis', 'acute_renal_fail', 'ckd', 'uti', 'calculus_urinary', 'other_kidney',
                'other_bladder', 'genitourinary_symp', 'prostate_hyp', 'male_genital_inflam', 'other_male_genital',
                'nonmalig_breast', 'inflam_fem_pelvic', 'endometriosis', 'prolapse_fem_gen', 'menstrual',
                'ovarian_cyst', 'menopausal', 'fem_infert', 'other_fem_genital', 'contraceptive_mgmt', 'spont_abortion',
                'induce_abortion', 'postabort_comp', 'ectopic_preg', 'other_comp_preg', 'hemorrhage_preg',
                'htn_comp_preg', 'early_labor', 'prolong_preg', 'dm_comp_preg', 'malposition', 'fetopelvic_disrupt',
                'prev_c_sect', 'fetal_distress', 'polyhydramnios', 'umbilical_comp', 'ob_trauma', 'forceps_deliv',
                'other_comp_birth', 'other_preg_deliv', 'skin_tissue_infec', 'other_skin_inflam', 'chronic_skin_ulcer',
                'other_skin', 'infec_arthritis', 'rheum_arth', 'osteo_arth', 'other_joint', 'spondylosis',
                'osteoporosis', 'pathological_fract', 'acq_foot_deform', 'other_acq_deform', 'systemic_lupus',
                'other_connective', 'other_bone_disease', 'cardiac_congen_anom', 'digest_congen_anom',
                'genito_congen_anom', 'ns_congen_anom', 'other_congen_anom', 'liveborn', 'short_gest',
                'intrauter_hypoxia', 'resp_distress_synd', 'hemolytic_jaundice', 'birth_trauma', 'other_perinatal',
                'joint_trauma', 'fract_femur_neck', 'spinal_cord', 'skull_face_fract', 'upper_limb_fract',
                'lower_limb_fract', 'other_fract', 'sprain_strain', 'intracranial', 'crush_injury', 'open_wound_head',
                'open_wound_extr', 'comp_of_device', 'comp_surg_proc', 'superficial_inj', 'burns', 'poison_psycho',
                'poison_other_med', 'poison_nonmed', 'other_ext_injury', 'syncope', 'fever_unknown', 'lymphadenitis',
                'gangrene', 'shock', 'naus_vom', 'abdominal_pain', 'malaise_fatigue', 'allergy', 'rehab_care',
                'admin_admiss', 'medical_eval', 'other_aftercare', 'other_screen', 'residual_codes', 'adjustment',
                'anxiety', 'adhd', 'dementia', 'develop_dis', 'child_disorder', 'impule_control', 'mood', 'personality',
                'schizo', 'alcohol', 'substance', 'suicide', 'mental_screen', 'misc_mental', 'e_cut_pierce', 'e_drown',
                'e_fall', 'e_fire', 'e_firearm', 'e_machine', 'e_mvt', 'e_cyclist', 'e_pedestrian', 'e_transport',
                'e_natural', 'e_overexert', 'e_poison', 'e_struckby', 'e_suffocate', 'e_ae_med_care', 'e_ae_med_drug',
                'e_other_class', 'e_other_nec', 'e_unspecified', 'e_place', 'age', 'sex', 'psych_hosp', 'ed_visit',
                'inpt_visit', 'outpt_visit', 'amb_visit']

    if full_model is False:
        # Applies restrictions to data set based on age, visit count, date differential, and mental illness
        # diagnoses
        if primary_dx is False:
            df = df.loc[
                (df['age'] >= 18) & (df['visit_count'] >= 2) & (df['date_diff'] >= 1)]  # Primary & Secondary Dx
        else:
            df = df.loc[(df['age'] >= 18) & (df['visit_count'] >= 2) & (df['date_diff'] >= 1) & (
                    df['enc'] != 2)]  # Primary Dx Only

        df = df.loc[(df['mood'] >= 1) | (df['schizo'] >= 1)]
    else:
        if primary_dx is True:
            df = df.loc[(df['enc'] != 2)]

    df2 = df.loc[:, features]
    df2 = df2.merge(drugs, on='person_id', how='left').fillna(0)  # Merge Outpatient Drugs df
    df2 = df2.merge(procs, on='person_id', how='left').fillna(0)  # Merge ICD-9/10 Procs df
    # cdrn.drop(cdrn.columns[cdrn.std() == 0], axis =1, inplace=True) # Drop columns with a STD == 0
    df2 = df2.drop(df2.columns[0], axis=1)

    # to_normalize = []
    # to_remain = []
    # cols = list(df2.columns.values)
    # for col in cols:
    #     if df2[col].max() > 1):
    #         to_normalize.append(col)
    #     else: to_remain.append(col)
    # to_normalize.append('person_id')
    #
    # df3 = df2.loc[:, to_normalize]
    # normalized_df = preprocessing.normalize(df3[:,1:])
    df2.to_csv(PATH + "cdrn_ccs_modified_label2_drugs_procdure_added.csv", sep=',')
    # sys.exit()
    X = df2.loc[:, df2.columns != 'psych_hosp']
    y = df2.loc[:, df2.columns == 'psych_hosp']

    X['random'] = np.random.random(
        size=len(X))  # creates 'random' variable to test feature selection; should be low importance

    return X, y


def subsample_df(x1, y1):
    case = y1.loc[y1.psych_hosp == 1]
    ctrl = y1.loc[y1.psych_hosp != 1]

    ctrl_sample = ctrl.sample(n=len(case))
    sub_df = case.append(ctrl_sample)
    sub_df_ = pd.merge(sub_df, x1, left_index=True, right_index=True)

    X_train = sub_df_.loc[:, sub_df_.columns != 'psych_hosp']
    y_train = sub_df_.loc[:, sub_df_.columns == 'psych_hosp']
    print('Subsampling Value Counts:\n', y_train.psych_hosp.value_counts(), '\n')

    return X_train, y_train


def smote_sample(x1, y1):
    os = SMOTE(random_state=SEED)
    columns = x1.columns

    X_train, y_train = os.fit_sample(x1, np.ravel(y1))
    X_train = pd.DataFrame(data=X_train, columns=columns)
    y_train = pd.DataFrame(data=y_train, columns=['psych_hosp'])
    print('SMOTE Sampling Value Counts:\n', y_train.psych_hosp.value_counts(), '\n')

    return X_train, y_train


def lr_model(x1, y1):
    X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.3, random_state=SEED)

    # Down-sample controls in training set, [1:1] case:control
    if subsample is True:
        X_train, y_train = subsample_df(X_train, y_train)
    # Implement SMOTE to balance training set, [1:1] case:control
    if smote is True:
        X_train, y_train = smote_sample(X_train, y_train)

    # Create regularization penalty space
    penalty = ['l1', 'l2']
    # Create regularization hyperparameter space
    C = np.logspace(0, 4, 10)
    # Create hyperparameter options
    weight = [None, 'balanced']
    hyperparameters = dict(C=C, penalty=penalty, class_weight=weight)

    # Create a base model
    logreg = LogisticRegression()
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=logreg, param_grid=hyperparameters,
                               cv=5, n_jobs=-1, verbose=1, refit=True, return_train_score=True)

    grid_search.fit(X_train, np.ravel(y_train))
    print("== LR Best Params ==", grid_search.best_params_, '\n')
    print('Mean test score: {}'.format(grid_search.cv_results_['mean_test_score']))
    print('Mean train score: {}\n'.format(grid_search.cv_results_['mean_train_score']))

    # logreg = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', class_weight='balanced', random_state=SEED)
    # lr_cv_score = cross_val_score(logreg, X_train, np.ravel(y_train), cv=10, scoring='roc_auc')
    gs = grid_search.best_estimator_
    gs.fit(X_train, np.ravel(y_train))

    logreg_predict = gs.predict(X_test)

    # print("=== All AUC Scores [CV - Train] ===")
    # print(lr_cv_score, '\n')
    # print("=== Mean AUC Score [CV - Train] ===")
    # print(lr_cv_score.mean(), '\n')
    print("=== Confusion Matrix [Test] ===")
    print(confusion_matrix(y_test, logreg_predict), '\n')
    print("=== Classification Report [Test] ===")
    print(classification_report(y_test, logreg_predict), '\n')
    print("=== AUC Score [Test] ===")
    print(roc_auc_score(y_test, logreg_predict), '\n')

    lr_roc_auc = roc_auc_score(y_test, logreg_predict)
    fpr, tpr, thresholds = roc_curve(y_test, gs.predict_proba(X_test)[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, label='LR Classifier (area = %0.3f)' % lr_roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic: Clinical Data Only [LR]')
    plt.legend(loc="lower right")
    plt.savefig(PATH2 + 'ROC_Psych_LR_' + outfile)
    plt.savefig(PATH + 'ROC_Psych_LR_' + outfile)
    plt.show()


def xgb_model(x1, y1):
    X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.3, random_state=SEED)

    # Down-sample controls in training set, [1:1] case:control
    if subsample is True:
        X_train, y_train = subsample_df(X_train, y_train)
    # Implement SMOTE to balance training set, [1:1] case:control
    if smote is True:
        X_train, y_train = smote_sample(X_train, y_train)

    columns = X_train.columns

    # Weight Rescale
    ratio = float(np.sum(y_train['psych_hosp'].values == 0) / np.sum(y_train['psych_hosp'].values == 1))

    # Instantiate the XGBClassifier and specify parameters
    xgb1 = XGBClassifier(
        learning_rate=0.1,
        n_estimators=500,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=ratio,
        seed=SEED)

    xgb_param = xgb1.get_xgb_params()
    xgtrain = xgb.DMatrix(X_train[columns].values, label=y_train['psych_hosp'].values)
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=xgb1.get_params()['n_estimators'], nfold=5,
                      metrics='auc', early_stopping_rounds=50)
    xgb1.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    xgb1.fit(X_train, np.ravel(y_train), eval_metric='auc')

    imp = importances(xgb1, X_test, y_test)  # permutation
    imp = imp.reset_index()

    unwanted = []
    feats = []
    for j, k in zip(imp['Feature'], imp['Importance']):
        if k <= 0:
            unwanted.append(j)
        else: feats.append(j)
    print('XGB Features:\n', feats, '\n')

    X_train = X_train.loc[:, feats]
    X_test = X_test.loc[:, feats]

    columns = X_train.columns

    # Instantiate the XGBClassifier and specify parameters
    xgb1 = XGBClassifier(
        learning_rate=0.1,
        n_estimators=500,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=ratio,
        seed=SEED)

    xgb_param = xgb1.get_xgb_params()
    xgtrain = xgb.DMatrix(X_train[columns].values, label=y_train['psych_hosp'].values)
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=xgb1.get_params()['n_estimators'], nfold=5,
                      metrics='auc', early_stopping_rounds=50)
    xgb1.set_params(n_estimators=cvresult.shape[0])
    xgb_cv_score = cross_val_score(xgb1, X_train, np.ravel(y_train), cv=10, scoring='roc_auc')

    # Fit the algorithm on the data
    xgb1.fit(X_train, np.ravel(y_train), eval_metric='auc')

    D = feature_dependence_matrix(X_train)
    viz1 = plot_dependence_heatmap(D, figsize=(11, 10))
    viz1.save(PATH + 'Psych_XGB_feat_depend_' + outfile)

    xgb_predict = xgb1.predict(X_test)

    print("=== All AUC Scores [CV - Train] ===")
    print(xgb_cv_score, '\n')
    print("=== Mean AUC Score [CV - Train] ===")
    print(xgb_cv_score.mean(), '\n')
    print("=== Confusion Matrix [Test] ===")
    print(confusion_matrix(y_test, xgb_predict), '\n')
    print("=== Classification Report [Test] ===")
    print(classification_report(y_test, xgb_predict), '\n')
    print("=== AUC Score [Test] ===")
    print(roc_auc_score(y_test, xgb_predict), '\n')

    imp = importances(xgb1, X_test, y_test)  # permutation
    viz2 = plot_importances(imp)
    viz2.save(PATH + 'Psych_XGB_feat_imp_' + outfile)
    imp = imp.reset_index()
    imp_ = imp[imp['Importance'] < 0.00000]


    xgb_roc_auc = roc_auc_score(y_test, xgb_predict)
    fpr, tpr, thresholds = roc_curve(y_test, xgb1.predict_proba(X_test)[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, label='XGB Classifier (area = %0.3f)' % xgb_roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic: Clinical Data Only [XGB]')
    plt.legend(loc="lower right")
    plt.savefig(PATH2 + 'ROC_Psych_XGB_' + outfile)
    plt.savefig(PATH + 'ROC_Psych_XGB_' + outfile)
    plt.show()

    return imp, feats


def rf_model(x1, y1):
    X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.3, random_state=SEED)

    # Down-sample controls in training set, [1:1] case:control
    if subsample is True:
        X_train, y_train = subsample_df(X_train, y_train)
    # Implement SMOTE to balance training set, [1:1] case:control
    if smote is True:
        X_train, y_train = smote_sample(X_train, y_train)

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    weights = [None, 'balanced']
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap,
                   'class_weight': weights}
    print(random_grid)

    rf = RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=1,
                                   random_state=SEED, n_jobs=-1, refit=True, return_train_score=True)
    # Fit the random search model
    rf_random.fit(X_train, np.ravel(y_train))
    print("== RF Best Params ==", rf_random.best_params_, '\n')

    # Selects the best performing estimator from the Randomized Search
    rf_best = rf_random.best_estimator_
    # Fit the new RF Estimator
    rf_best.fit(X_train, np.ravel(y_train))

    # Determine Feature Importances by Permutation
    imp = importances(rf_best, X_test, y_test)  # permutation
    imp = imp.reset_index()

    # Create a list of Features from the importance output
    unwanted = []
    feats = []
    for j, k in zip(imp['Feature'], imp['Importance']):
        if k <= 0:
            unwanted.append(j)
        else:
            feats.append(j)
    print('RF Features:\n', feats, '\n')

    X_train = X_train.loc[:, feats]
    X_test = X_test.loc[:, feats]

    # Instantiate a new Random Forest Classifier
    rf = RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random2 = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=1,
                                   random_state=SEED, n_jobs=-1, refit=True, return_train_score=True)
    # Fit the random search model
    rf_random2.fit(X_train, np.ravel(y_train))
    print("== RF Best Params ==", rf_random2.best_params_, '\n')
    print('Mean test score: {}'.format(rf_random2.cv_results_['mean_test_score']))
    print('Mean train score: {}\n'.format(rf_random2.cv_results_['mean_train_score']))

    rf_best = rf_random2.best_estimator_
    rf_best.fit(X_train, np.ravel(y_train))

    D = feature_dependence_matrix(X_train)
    viz1 = plot_dependence_heatmap(D, figsize=(11, 10))
    viz1.save(PATH + 'Psych_RF_feat_depend_' + outfile)
    imp = importances(rf_best, X_test, y_test)  # permutation
    imp = imp.reset_index()
    imp_ = imp[imp['Importance'] <= 0.00000]

    feats = []
    for _ in imp_['Feature']:
        feats.append(_)

    rf_predict = rf_best.predict(X_test)

    print("=== Confusion Matrix [Test] ===")
    print(confusion_matrix(y_test, rf_predict), '\n')
    print("=== Classification Report [Test] ===")
    print(classification_report(y_test, rf_predict), '\n')
    print("=== AUC Score [Test] ===")
    print(roc_auc_score(y_test, rf_predict), '\n')

    imp = importances(rf_best, X_test, y_test)  # permutation
    viz2 = plot_importances(imp)
    viz2.save(PATH + 'Psych_RF_feat_imp_' + outfile)

    rf_roc_auc = roc_auc_score(y_test, rf_predict)
    fpr, tpr, thresholds = roc_curve(y_test, rf_best.predict_proba(X_test)[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, label='RF Classifier (area = %0.3f)' % rf_roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic: Clinical Data Only [RF]')
    plt.legend(loc="lower right")
    plt.savefig(PATH2 + 'ROC_Psych_RF_' + outfile)
    plt.savefig(PATH + 'ROC_Psych_RF_' + outfile)
    plt.show()

    return imp, feats


def mlp_model(x1, y1):
    X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.3, random_state=SEED)

    # Down-sample controls in training set, [1:1] case:control
    if subsample is True:
        X_train, y_train = subsample_df(X_train, y_train)
    # Implement SMOTE to balance training set, [1:1] case:control
    if smote is True:
        X_train, y_train = smote_sample(X_train, y_train)

    mlp = MLPClassifier(hidden_layer_sizes=((40,)*5), activation='logistic', max_iter=1000, random_state=SEED, learning_rate='constant', early_stopping=False, alpha=0.01)
    mlp.fit(X_train, np.ravel(y_train))

    mlp_predict = mlp.predict(X_test)

    print("=== Confusion Matrix [Test] ===")
    print(confusion_matrix(y_test, mlp_predict), '\n')
    print("=== Classification Report [Test] ===")
    print(classification_report(y_test, mlp_predict), '\n')
    print("=== AUC Score [Test] ===")
    print(roc_auc_score(y_test, mlp_predict), '\n')

    mlp_roc_auc = roc_auc_score(y_test, mlp_predict)
    fpr, tpr, thresholds = roc_curve(y_test, mlp.predict_proba(X_test)[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, label='MLP Classifier (area = %0.3f)' % mlp_roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic: Clinical Data Only [MLP]')
    plt.legend(loc="lower right")
    plt.savefig(PATH2 + 'ROC_Psych_MLP_' + outfile)
    plt.savefig(PATH + 'ROC_Psych_MLP_' + outfile)
    plt.show()


X, y = data_processing(cdrn)

lr_model(X, y)

xgb_imp, xgb_feats = xgb_model(X, y)

rf_imp, rf_feats = rf_model(X, y)

mlp_model(X, y)
