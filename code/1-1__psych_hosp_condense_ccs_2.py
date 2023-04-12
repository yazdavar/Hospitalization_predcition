'''
File to connect to SQL data based and read relvant table
Author: Amir Yazdavar
'''


import pandas as pd
import datetime as dt
PATH = '/home/ubuntu/amir/depression_sdh/depression_sdh/Joe_results/'

# IMPORT DATA
case_control = pd.read_csv(PATH+'cdrn_psych_casecontrol2.csv',encoding='utf-8',delimiter="\t")

# CONDENSE DIAGNOSIS DATA TO COUNTS, GROUPBY PATIENT_ID
def dx_condense1(df):
    output_filepath = PATH+'cdrn_ccs_grouped1.csv'

    cdrn_ccs_results = pd.read_csv(PATH+'cdrn_ccs_results1_.csv',delimiter="\t",
                                   encoding='utf-8')
    print("Number of unique person_id in cdrn_ccs_results: ", cdrn_ccs_results.person_id.nunique())

    df = df.merge(cdrn_ccs_results, on='person_id', how='left')
    df = df[(df.condition_start_date >= df.one_year_date) & (df.condition_start_date < df.index_date)]

    # Converts Longitudinal Data to Aggregate Counts Per Patient
    cdrn_ccs_grouped = df.groupby('person_id').agg(
        {"tuberculosis": sum, "septic": sum, "bact_infec": sum,
         "mycoses": sum, "hiv": sum, "hepatitis": sum, "viral_infec": sum, "other_infec": sum, "sti": sum,
         "screen_infec": sum, "head_ca": sum, "esophagus_ca": sum, "stomach_ca": sum, "colon_ca": sum,
         "rectum_ca": sum, "liver_ca": sum, "pancreas_ca": sum, "gi_ca": sum, "lung_ca": sum, "resp_ca": sum,
         "bone_ca": sum, "melanoma": sum, "nonepi_skin_ca": sum, "breast_ca": sum, "uterus_ca": sum,
         "cervix_ca": sum, "ovary_ca": sum, "fem_genital_ca": sum, "prostate_ca": sum, "testes_ca": sum,
         "male_genital_ca": sum, "bladder_ca": sum, "kidney_ca": sum, "urinary_ca": sum, "brain_ca": sum,
         "thyroid_ca": sum, "hodgkins_lymph": sum, "non_hodgkins_lymph": sum, "leukemia": sum,
         "mult_myeloma": sum, "other_ca": sum, "secndry_malig": sum, "malig_neoplasm": sum,
         "neoplasm_unspec": sum, "maint_chemo": sum, "ben_neoplasm_uterus": sum, "other_ben_neoplasm": sum,
         "thyroid": sum, "dm_wo_comp": sum, "dm_w_comp": sum, "other_endocrine": sum, "nutrition": sum,
         "lipid_metabo": sum, "gout": sum, "fluid_electrolyte": sum, "cyst_fibrosis": sum, "immunity": sum,
         "other_metabo": sum, "other_anemia": sum, "post_hemorr_anemia": sum, "sickle_cell": sum,
         "coag_anemia": sum, "wbc_disease": sum, "other_heme": sum, "meningitis_notb": sum,
         "encephalitis_notb": sum, "other_cns": sum, "parkinsons": sum})

    print("Patients in CCS Grouped 1: {}".format(len(cdrn_ccs_grouped)))
    pd.DataFrame(cdrn_ccs_grouped).to_csv(output_filepath, encoding='utf-8')
    print("Finished condensing CCS Results 1\n")
    


def dx_condense2(df):
    output_filepath = PATH+'cdrn_ccs_grouped2.csv'

    cdrn_ccs_results = pd.read_csv(PATH+'cdrn_ccs_results2_.csv' ,delimiter="\t",
                                   encoding='utf-8')
    print("Number of unique person_id in cdrn_ccs_results: ", cdrn_ccs_results.person_id.nunique())

    df = df.merge(cdrn_ccs_results, on='person_id', how='left')
    df = df[(df.condition_start_date >= df.one_year_date) & (df.condition_start_date < df.index_date)]

    # Converts Longitudinal Data to Aggregate Counts Per Patient
    cdrn_ccs_grouped = df.groupby('person_id').agg(
        {"mult_scler": sum, "other_hered_degen": sum, "paralysis": sum, "epilepsy": sum, "headache": sum, "coma": sum,
         "cataract": sum, "retinopathy": sum, "glaucoma": sum, "blindness": sum, "eye_inflam": sum, "other_eye": sum,
         "otitis_media": sum, "dizzy": sum, "other_ear_sense": sum, "other_ns_disorder": sum,
         "heart_valve": sum, "peri_endo_carditis": sum, "essential_htn": sum, "htn_w_comp": sum, "acute_mi": sum,
         "coronary_athero": sum, "chest_pain_nos": sum, "pulmonary_hd": sum, "other_heart_disease": sum,
         "conduction": sum, "cardiac_dysrhythm": sum, "cardiac_arrest": sum, "chf": sum, "acute_cvd": sum,
         "occlu_cereb_artery": sum, "other_cvd": sum, "tran_cereb_isch": sum, "late_effect_cvd": sum, "pvd": sum,
         "artery_aneurysm": sum, "artery_embolism": sum, "other_circ": sum, "phlebitis": sum,
         "varicose_vein": sum, "hemorrhoid": sum, "other_vein_lymph": sum, "pneumonia": sum, "influenza": sum,
         "acute_tonsil": sum, "acute_bronch": sum, "upper_resp_infec": sum, "copd": sum, "asthma": sum,
         "asp_pneumonitis": sum, "pneumothorax": sum, "resp_failure": sum, "lung_disease": sum,
         "other_low_resp": sum, "other_up_resp": sum, "intestinal_infec": sum, "teeth_jaw": sum,
         "mouth_disease": sum, "esophagus": sum, "gastro_ulcer": sum, "gastritis": sum, "other_stomach_duo": sum})

    print("Patients in CCS Grouped 2: {}".format(len(cdrn_ccs_grouped)))
    pd.DataFrame(cdrn_ccs_grouped).to_csv(output_filepath, encoding='utf-8')
    print("Finished condensing CCS Results 2\n")
    


def dx_condense3(df):
    output_filepath = PATH+'cdrn_ccs_grouped3.csv'

    cdrn_ccs_results = pd.read_csv(PATH+'cdrn_ccs_results3_.csv',delimiter="\t",
                                   encoding='utf-8')
    print("Number of unique person_id in cdrn_ccs_results: ", cdrn_ccs_results.person_id.nunique())

    df = df.merge(cdrn_ccs_results, on='person_id', how='left')
    df = df[(df.condition_start_date >= df.one_year_date) & (df.condition_start_date < df.index_date)]

    # Converts Longitudinal Data to Aggregate Counts Per Patient
    cdrn_ccs_grouped = df.groupby('person_id').agg(
        {"appendicitis": sum, "hernia_abd": sum, "regional_enteriritis": sum, "intestinal_obstruct": sum,
         "diverticulitis": sum, "anal_condition": sum, "peritonitis": sum, "biliary_tract": sum,
         "other_liver": sum, "pancreatic": sum, "gastro_hemorrhage": sum, "noninfec_gastro": sum,
         "other_gastro": sum, "nephritis": sum, "acute_renal_fail": sum, "ckd": sum, "uti": sum,
         "calculus_urinary": sum, "other_kidney": sum, "other_bladder": sum, "genitourinary_symp": sum,
         "prostate_hyp": sum, "male_genital_inflam": sum, "other_male_genital": sum, "nonmalig_breast": sum,
         "inflam_fem_pelvic": sum, "endometriosis": sum, "prolapse_fem_gen": sum, "menstrual": sum,
         "ovarian_cyst": sum, "menopausal": sum, "fem_infert": sum, "other_fem_genital": sum,
         "contraceptive_mgmt": sum, "spont_abortion": sum, "induce_abortion": sum, "postabort_comp": sum,
         "ectopic_preg": sum, "other_comp_preg": sum, "hemorrhage_preg": sum, "htn_comp_preg": sum,
         "early_labor": sum, "prolong_preg": sum, "dm_comp_preg": sum, "malposition": sum,
         "fetopelvic_disrupt": sum, "prev_c_sect": sum, "fetal_distress": sum, "polyhydramnios": sum,
         "umbilical_comp": sum, "ob_trauma": sum, "forceps_deliv": sum, "other_comp_birth": sum,
         "other_preg_deliv": sum, "skin_tissue_infec": sum, "other_skin_inflam": sum, "chronic_skin_ulcer": sum,
         "other_skin": sum, "infec_arthritis": sum, "rheum_arth": sum, "osteo_arth": sum, "other_joint": sum,
         "spondylosis": sum, "osteoporosis": sum, "pathological_fract": sum, "acq_foot_deform": sum,
         "other_acq_deform": sum, "systemic_lupus": sum, "other_connective": sum, "other_bone_disease": sum,
         "cardiac_congen_anom": sum, "digest_congen_anom": sum, "genito_congen_anom": sum, "ns_congen_anom": sum,
         "other_congen_anom": sum, "liveborn": sum, "short_gest": sum, "intrauter_hypoxia": sum,
         "resp_distress_synd": sum, "hemolytic_jaundice": sum})

    print("Patients in CCS Grouped 3: {}".format(len(cdrn_ccs_grouped)))
    pd.DataFrame(cdrn_ccs_grouped).to_csv(output_filepath, encoding='utf-8')
    print("Finished condensing CCS Results 3\n")
    


def dx_condense4(df):
    output_filepath = PATH+'cdrn_ccs_grouped4.csv'

    cdrn_ccs_results = pd.read_csv(PATH+'cdrn_ccs_results4_.csv',delimiter="\t",
                                   encoding='utf-8')
    print("Number of unique person_id in cdrn_ccs_results: ", cdrn_ccs_results.person_id.nunique())

    df = df.merge(cdrn_ccs_results, on='person_id', how='left')
    df = df[(df.condition_start_date >= df.one_year_date) & (df.condition_start_date < df.index_date)]

    # Converts Longitudinal Data to Aggregate Counts Per Patient
    cdrn_ccs_grouped = df.groupby('person_id').agg(
        {"birth_trauma": sum, "other_perinatal": sum, "joint_trauma": sum, "fract_femur_neck": sum, "spinal_cord": sum,
         "skull_face_fract": sum, "upper_limb_fract": sum, "lower_limb_fract": sum, "other_fract": sum,
         "sprain_strain": sum, "intracranial": sum, "crush_injury": sum, "open_wound_head": sum, "open_wound_extr": sum,
         "comp_of_device": sum, "comp_surg_proc": sum, "superficial_inj": sum, "burns": sum, "poison_psycho": sum,
         "poison_other_med": sum, "poison_nonmed": sum, "other_ext_injury": sum, "syncope": sum, "fever_unknown": sum,
         "lymphadenitis": sum, "gangrene": sum, "shock": sum, "naus_vom": sum, "abdominal_pain": sum,
         "malaise_fatigue": sum, "allergy": sum, "rehab_care": sum, "admin_admiss": sum, "medical_eval": sum,
         "other_aftercare": sum, "other_screen": sum, "residual_codes": sum, "adjustment": sum, "anxiety": sum,
         "adhd": sum, "dementia": sum, "develop_dis": sum, "child_disorder": sum, "impule_control": sum, "mood": sum,
         "personality": sum, "schizo": sum, "alcohol": sum, "substance": sum, "suicide": sum, "mental_screen": sum,
         "misc_mental": sum, "e_cut_pierce": sum, "e_drown": sum, "e_fall": sum, "e_fire": sum, "e_firearm": sum,
         "e_machine": sum, "e_mvt": sum, "e_cyclist": sum, "e_pedestrian": sum, "e_transport": sum, "e_natural": sum,
         "e_overexert": sum, "e_poison": sum, "e_struckby": sum, "e_suffocate": sum, "e_ae_med_care": sum,
         "e_ae_med_drug": sum, "e_other_class": sum, "e_other_nec": sum, "e_unspecified": sum, "e_place": sum})

    print("Patients in CCS Grouped 4: {}".format(len(cdrn_ccs_grouped)))
    pd.DataFrame(cdrn_ccs_grouped).to_csv(output_filepath, encoding='utf-8')
    print("Finished condensing CCS Results 4\n")
    


#  CREATE HEALTH CARE UTILIZATION COUNTS
def utilization(df):
    output_filepath = PATH+'psych_casecontrol_visit.csv'
    visits = pd.read_csv(PATH+'cdrn_all_visits_1218.csv',
                         encoding='utf-8', names=['person_id', 'visit_occurrence_id', 'visit_start_date',
                                                  'visit_concept_id'],delimiter="\t")
    df = df.merge(visits, on='person_id', how='left')
    df1 = df[(df.visit_start_date >= df.one_year_date) & (df.visit_start_date < df.index_date)].copy()
    df1['labels'] = df1.loc[:,'visit_concept_id']

    features = ['person_id', 'visit_concept_id', 'labels']
    df1 = df1.loc[:, features]

    df2 = pd.pivot_table(df1, values='visit_concept_id', index='person_id', columns='labels', aggfunc='count').fillna(0)

    df2.rename(columns={0:'missing', 9203: 'ed_visit', 9201: 'inpt_visit', 9202: 'outpt_visit', 44814711: 'amb_visit',
                        44814649: 'other_visit'}, inplace=True)
    df2 = df2.reset_index()

    features2 = ['person_id', 'ed_visit', 'inpt_visit', 'outpt_visit', 'amb_visit', 'other_visit']
    df2 = df2.loc[:, features2]

    pd.DataFrame(df2).to_csv(output_filepath, encoding='utf-8')


#dx_condense1(case_control)


# dx_condense2(case_control)
# print ("done2")

# dx_condense3(case_control)
# print ("done3")

# dx_condense4(case_control)
# print ("done4")

utilization(case_control)
