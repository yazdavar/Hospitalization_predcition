'''
File to connect to SQL data based and read relvant table
Author: Amir Yazdavar
'''


import pandas as pd
import sys
from functools import reduce
PATH = '/home/ubuntu/amir/depression_sdh/depression_sdh/Joe_results/'

# IMPORT DATA
case_control = pd.read_csv(PATH+'cdrn_psych_casecontrol2.csv', delimiter="\t",
                      encoding='utf-8')

# COMBINE GROUPED DIAGNOSIS DATA
def ccs_aggregate(df):
    cdrn_ccs_grouped1 = pd.read_csv(PATH+'cdrn_ccs_grouped1.csv',encoding='utf-8')
    print("cdrn_ccs_grouped length1: {}".format(len(cdrn_ccs_grouped1)))
    cdrn_ccs_grouped2 = pd.read_csv(PATH+'cdrn_ccs_grouped2.csv',encoding='utf-8')
    print("cdrn_ccs_grouped length2: {}".format(len(cdrn_ccs_grouped2)))
    cdrn_ccs_grouped3 = pd.read_csv(PATH+'cdrn_ccs_grouped3.csv',encoding='utf-8')
    print("cdrn_ccs_grouped length3: {}".format(len(cdrn_ccs_grouped3)))
    cdrn_ccs_grouped4 = pd.read_csv(PATH+'cdrn_ccs_grouped4.csv',encoding='utf-8')
    print("cdrn_ccs_grouped length4: {}".format(len(cdrn_ccs_grouped4)))


    dfs = [cdrn_ccs_grouped1, cdrn_ccs_grouped2, cdrn_ccs_grouped3, cdrn_ccs_grouped4]
    cdrn_ccs_grouped = reduce(lambda left, right: pd.merge(left, right, on='person_id'), dfs)
    print("Length CCS Grouped: {}".format(len(cdrn_ccs_grouped)))

    return cdrn_ccs_grouped


# Iam not sure this is the best representation
# CONVERT DX TO COUNT / BINARY FEATURES
def dx_convert(df):
    # output_filepath =  PATH+'cdrn_ccs_modified.csv'

    visits = pd.read_csv(PATH+'psych_casecontrol_visit.csv',
                         encoding='utf-8')

    ccs_modified = {"person_id": [], "tuberculosis": [], "septic": [], "bact_infec": [], "mycoses": [], "hiv": [],
                    "hepatitis": [], "viral_infec": [], "other_infec": [], "sti": [], "screen_infec": [], "head_ca": [],
                    "esophagus_ca": [], "stomach_ca": [], "colon_ca": [], "rectum_ca": [], "liver_ca": [],
                    "pancreas_ca": [], "gi_ca": [], "lung_ca": [], "resp_ca": [], "bone_ca": [], "melanoma": [],
                    "nonepi_skin_ca": [], "breast_ca": [], "uterus_ca": [], "cervix_ca": [], "ovary_ca": [],
                    "fem_genital_ca": [], "prostate_ca": [], "testes_ca": [], "male_genital_ca": [], "bladder_ca": [],
                    "kidney_ca": [], "urinary_ca": [], "brain_ca": [], "thyroid_ca": [], "hodgkins_lymph": [],
                    "non_hodgkins_lymph": [], "leukemia": [], "mult_myeloma": [], "other_ca": [], "secndry_malig": [],
                    "malig_neoplasm": [], "neoplasm_unspec": [], "maint_chemo": [], "ben_neoplasm_uterus": [],
                    "other_ben_neoplasm": [], "thyroid": [], "dm_wo_comp": [], "dm_w_comp": [], "other_endocrine": [],
                    "nutrition": [], "lipid_metabo": [], "gout": [], "fluid_electrolyte": [], "cyst_fibrosis": [],
                    "immunity": [], "other_metabo": [], "other_anemia": [], "post_hemorr_anemia": [], "sickle_cell": [],
                    "coag_anemia": [], "wbc_disease": [], "other_heme": [], "meningitis_notb": [],
                    "encephalitis_notb": [], "other_cns": [], "parkinsons": [], "mult_scler": [],
                    "other_hered_degen": [], "paralysis": [], "epilepsy": [], "headache": [], "coma": [],
                    "cataract": [], "retinopathy": [], "glaucoma": [], "blindness": [], "eye_inflam": [],
                    "other_eye": [], "otitis_media": [], "dizzy": [], "other_ear_sense": [], "other_ns_disorder": [],
                    "heart_valve": [], "peri_endo_carditis": [], "essential_htn": [], "htn_w_comp": [], "acute_mi": [],
                    "coronary_athero": [], "chest_pain_nos": [], "pulmonary_hd": [], "other_heart_disease": [],
                    "conduction": [], "cardiac_dysrhythm": [], "cardiac_arrest": [], "chf": [], "acute_cvd": [],
                    "occlu_cereb_artery": [], "other_cvd": [], "tran_cereb_isch": [], "late_effect_cvd": [], "pvd": [],
                    "artery_aneurysm": [], "artery_embolism": [], "other_circ": [], "phlebitis": [],
                    "varicose_vein": [], "hemorrhoid": [], "other_vein_lymph": [], "pneumonia": [], "influenza": [],
                    "acute_tonsil": [], "acute_bronch": [], "upper_resp_infec": [], "copd": [], "asthma": [],
                    "asp_pneumonitis": [], "pneumothorax": [], "resp_failure": [], "lung_disease": [],
                    "other_low_resp": [], "other_up_resp": [], "intestinal_infec": [], "teeth_jaw": [],
                    "mouth_disease": [], "esophagus": [], "gastro_ulcer": [], "gastritis": [], "other_stomach_duo": [],
                    "appendicitis": [], "hernia_abd": [], "regional_enteriritis": [], "intestinal_obstruct": [],
                    "diverticulitis": [], "anal_condition": [], "peritonitis": [], "biliary_tract": [],
                    "other_liver": [], "pancreatic": [], "gastro_hemorrhage": [], "noninfec_gastro": [],
                    "other_gastro": [], "nephritis": [], "acute_renal_fail": [], "ckd": [], "uti": [],
                    "calculus_urinary": [], "other_kidney": [], "other_bladder": [], "genitourinary_symp": [],
                    "prostate_hyp": [], "male_genital_inflam": [], "other_male_genital": [], "nonmalig_breast": [],
                    "inflam_fem_pelvic": [], "endometriosis": [], "prolapse_fem_gen": [], "menstrual": [],
                    "ovarian_cyst": [], "menopausal": [], "fem_infert": [], "other_fem_genital": [],
                    "contraceptive_mgmt": [], "spont_abortion": [], "induce_abortion": [], "postabort_comp": [],
                    "ectopic_preg": [], "other_comp_preg": [], "hemorrhage_preg": [], "htn_comp_preg": [],
                    "early_labor": [], "prolong_preg": [], "dm_comp_preg": [], "malposition": [],
                    "fetopelvic_disrupt": [], "prev_c_sect": [], "fetal_distress": [], "polyhydramnios": [],
                    "umbilical_comp": [], "ob_trauma": [], "forceps_deliv": [], "other_comp_birth": [],
                    "other_preg_deliv": [], "skin_tissue_infec": [], "other_skin_inflam": [], "chronic_skin_ulcer": [],
                    "other_skin": [], "infec_arthritis": [], "rheum_arth": [], "osteo_arth": [], "other_joint": [],
                    "spondylosis": [], "osteoporosis": [], "pathological_fract": [], "acq_foot_deform": [],
                    "other_acq_deform": [], "systemic_lupus": [], "other_connective": [], "other_bone_disease": [],
                    "cardiac_congen_anom": [], "digest_congen_anom": [], "genito_congen_anom": [], "ns_congen_anom": [],
                    "other_congen_anom": [], "liveborn": [], "short_gest": [], "intrauter_hypoxia": [],
                    "resp_distress_synd": [], "hemolytic_jaundice": [], "birth_trauma": [], "other_perinatal": [],
                    "joint_trauma": [], "fract_femur_neck": [], "spinal_cord": [], "skull_face_fract": [],
                    "upper_limb_fract": [], "lower_limb_fract": [], "other_fract": [], "sprain_strain": [],
                    "intracranial": [], "crush_injury": [], "open_wound_head": [], "open_wound_extr": [],
                    "comp_of_device": [], "comp_surg_proc": [], "superficial_inj": [], "burns": [], "poison_psycho": [],
                    "poison_other_med": [], "poison_nonmed": [], "other_ext_injury": [], "syncope": [],
                    "fever_unknown": [], "lymphadenitis": [], "gangrene": [], "shock": [], "naus_vom": [],
                    "abdominal_pain": [], "malaise_fatigue": [], "allergy": [], "rehab_care": [], "admin_admiss": [],
                    "medical_eval": [], "other_aftercare": [], "other_screen": [], "residual_codes": [],
                    "adjustment": [], "anxiety": [], "adhd": [], "dementia": [], "develop_dis": [],
                    "child_disorder": [], "impule_control": [], "mood": [], "personality": [], "schizo": [],
                    "alcohol": [], "substance": [], "suicide": [], "mental_screen": [], "misc_mental": [],
                    "e_cut_pierce": [], "e_drown": [], "e_fall": [], "e_fire": [], "e_firearm": [], "e_machine": [],
                    "e_mvt": [], "e_cyclist": [], "e_pedestrian": [], "e_transport": [], "e_natural": [],
                    "e_overexert": [], "e_poison": [], "e_struckby": [], "e_suffocate": [], "e_ae_med_care": [],
                    "e_ae_med_drug": [], "e_other_class": [], "e_other_nec": [], "e_unspecified": [], "e_place": []}

    # for _ in case_control['person_id']:
    #     if _ not in list(df['person_id']):
    #         ccs_modified['person_id'].append(_)
    #         for k, v in ccs_modified.items():
    #             if k != 'person_id':
    #                 v.append(0)

    for _ in df['person_id']:
        ccs_modified["person_id"].append(_)
    print(len(ccs_modified["person_id"]))

    for _ in df['tuberculosis']:
        if _ >= 1:
            ccs_modified["tuberculosis"].append(1)
        else: ccs_modified["tuberculosis"].append(0)
    print("Completed Recoding of: 'tuberculosis' ")

    for _ in df['septic']:
        ccs_modified["septic"].append(_)
    print("Completed Recoding of: 'septic' ")

    for _ in df['bact_infec']:
        ccs_modified["bact_infec"].append(_)
    print("Completed Recoding of: 'bact_infec' ")

    for _ in df['mycoses']:
        ccs_modified["mycoses"].append(_)
    print("Completed Recoding of: 'mycoses' ")

    for _ in df['hiv']:
        if _ >= 1:
            ccs_modified["hiv"].append(1)
        else: ccs_modified["hiv"].append(0)
    print("Completed Recoding of: 'hiv' ")

    for _ in df['hepatitis']:
        if _ >= 1:
            ccs_modified["hepatitis"].append(1)
        else: ccs_modified["hepatitis"].append(0)
    print("Completed Recoding of: 'hepatitis' ")

    for _ in df['viral_infec']:
        ccs_modified["viral_infec"].append(_)
    print("Completed Recoding of: 'viral_infec' ")

    for _ in df['other_infec']:
        ccs_modified["other_infec"].append(_)
    print("Completed Recoding of: 'other_infec' ")

    for _ in df['sti']:
        ccs_modified["sti"].append(_)
    print("Completed Recoding of: 'sti' ")

    for _ in df['screen_infec']:
        ccs_modified["screen_infec"].append(_)
    print("Completed Recoding of: 'screen_infec' ")

    for _ in df['head_ca']:
        ccs_modified["head_ca"].append(_)
    print("Completed Recoding of: 'head_ca' ")

    for _ in df['esophagus_ca']:
        ccs_modified["esophagus_ca"].append(_)
    print("Completed Recoding of: 'esophagus_ca' ")

    for _ in df['stomach_ca']:
        ccs_modified["stomach_ca"].append(_)
    print("Completed Recoding of: 'stomach_ca' ")

    for _ in df['colon_ca']:
        if _ >= 1:
            ccs_modified["colon_ca"].append(1)
        else: ccs_modified["colon_ca"].append(0)
    print("Completed Recoding of: 'colon_ca' ")

    for _ in df['rectum_ca']:
        if _ >= 1:
            ccs_modified["rectum_ca"].append(1)
        else: ccs_modified["rectum_ca"].append(0)
    print("Completed Recoding of: 'rectum_ca' ")

    for _ in df['liver_ca']:
        ccs_modified["liver_ca"].append(_)
    print("Completed Recoding of: 'liver_ca' ")

    for _ in df['pancreas_ca']:
        ccs_modified["pancreas_ca"].append(_)
    print("Completed Recoding of: 'pancreas_ca' ")

    for _ in df['gi_ca']:
        ccs_modified["gi_ca"].append(_)
    print("Completed Recoding of: 'gi_ca' ")

    for _ in df['lung_ca']:
        if _ >= 1:
            ccs_modified["lung_ca"].append(1)
        else: ccs_modified["lung_ca"].append(0)
    print("Completed Recoding of: 'lung_ca' ")

    for _ in df['resp_ca']:
        ccs_modified["resp_ca"].append(_)
    print("Completed Recoding of: 'resp_ca' ")

    for _ in df['bone_ca']:
        ccs_modified["bone_ca"].append(_)
    print("Completed Recoding of: 'bone_ca' ")

    for _ in df['melanoma']:
        ccs_modified["melanoma"].append(_)
    print("Completed Recoding of: 'melanoma' ")

    for _ in df['nonepi_skin_ca']:
        ccs_modified["nonepi_skin_ca"].append(_)
    print("Completed Recoding of: 'nonepi_skin_ca' ")

    for _ in df['breast_ca']:
        if _ >= 1:
            ccs_modified["breast_ca"].append(1)
        else: ccs_modified["breast_ca"].append(0)
    print("Completed Recoding of: 'breast_ca' ")

    for _ in df['uterus_ca']:
        if _ >= 1:
            ccs_modified["uterus_ca"].append(1)
        else: ccs_modified["uterus_ca"].append(0)
    print("Completed Recoding of: 'uterus_ca' ")

    for _ in df['cervix_ca']:
        ccs_modified["cervix_ca"].append(_)
    print("Completed Recoding of: 'cervix_ca' ")

    for _ in df['ovary_ca']:
        ccs_modified["ovary_ca"].append(_)
    print("Completed Recoding of: 'ovary_ca' ")

    for _ in df['fem_genital_ca']:
        ccs_modified["fem_genital_ca"].append(_)
    print("Completed Recoding of: 'fem_genital_ca' ")

    for _ in df['prostate_ca']:
        if _ >= 1:
            ccs_modified["prostate_ca"].append(1)
        else: ccs_modified["prostate_ca"].append(0)
    print("Completed Recoding of: 'prostate_ca' ")

    for _ in df['testes_ca']:
        ccs_modified["testes_ca"].append(_)
    print("Completed Recoding of: 'testes_ca' ")

    for _ in df['male_genital_ca']:
        ccs_modified["male_genital_ca"].append(_)
    print("Completed Recoding of: 'male_genital_ca' ")

    for _ in df['bladder_ca']:
        ccs_modified["bladder_ca"].append(_)
    print("Completed Recoding of: 'bladder_ca' ")

    for _ in df['kidney_ca']:
        if _ >= 1:
            ccs_modified["kidney_ca"].append(1)
        else: ccs_modified["kidney_ca"].append(0)
    print("Completed Recoding of: 'kidney_ca' ")

    for _ in df['urinary_ca']:
        if _ >= 1:
            ccs_modified["urinary_ca"].append(1)
        else: ccs_modified["urinary_ca"].append(0)
    print("Completed Recoding of: 'urinary_ca' ")

    for _ in df['brain_ca']:
        ccs_modified["brain_ca"].append(_)
    print("Completed Recoding of: 'brain_ca' ")

    for _ in df['thyroid_ca']:
        ccs_modified["thyroid_ca"].append(_)
    print("Completed Recoding of: 'thyroid_ca' ")

    for _ in df['hodgkins_lymph']:
        if _ >= 1:
            ccs_modified["hodgkins_lymph"].append(1)
        else: ccs_modified["hodgkins_lymph"].append(0)
    print("Completed Recoding of: 'hodgkins_lymph' ")

    for _ in df['non_hodgkins_lymph']:
        if _ >= 1:
            ccs_modified["non_hodgkins_lymph"].append(1)
        else: ccs_modified["non_hodgkins_lymph"].append(0)
    print("Completed Recoding of: 'non_hodgkins_lymph' ")

    for _ in df['leukemia']:
        if _ >= 1:
            ccs_modified["leukemia"].append(1)
        else: ccs_modified["leukemia"].append(0)
    print("Completed Recoding of: 'leukemia' ")

    for _ in df['mult_myeloma']:
        ccs_modified["mult_myeloma"].append(_)
    print("Completed Recoding of: 'mult_myeloma' ")

    for _ in df['other_ca']:
        ccs_modified["other_ca"].append(_)
    print("Completed Recoding of: 'other_ca' ")

    for _ in df['secndry_malig']:
        ccs_modified["secndry_malig"].append(_)
    print("Completed Recoding of: 'secndry_malig' ")

    for _ in df['malig_neoplasm']:
        ccs_modified["malig_neoplasm"].append(_)
    print("Completed Recoding of: 'malig_neoplasm' ")

    for _ in df['neoplasm_unspec']:
        ccs_modified["neoplasm_unspec"].append(_)
    print("Completed Recoding of: 'neoplasm_unspec' ")

    for _ in df['maint_chemo']:
        ccs_modified["maint_chemo"].append(_)
    print("Completed Recoding of: 'maint_chemo' ")

    for _ in df['ben_neoplasm_uterus']:
        ccs_modified["ben_neoplasm_uterus"].append(_)
    print("Completed Recoding of: 'ben_neoplasm_uterus' ")

    for _ in df['other_ben_neoplasm']:
        ccs_modified["other_ben_neoplasm"].append(_)
    print("Completed Recoding of: 'other_ben_neoplasm' ")

    for _ in df['thyroid']:
        if _ >= 1:
            ccs_modified["thyroid"].append(1)
        else: ccs_modified["thyroid"].append(0)
    print("Completed Recoding of: 'thyroid' ")

    for _ in df['dm_wo_comp']:
        if _ >= 1:
            ccs_modified["dm_wo_comp"].append(1)
        else: ccs_modified["dm_wo_comp"].append(0)
    print("Completed Recoding of: 'dm_wo_comp' ")

    for _ in df['dm_w_comp']:
        if _ >= 1:
            ccs_modified["dm_w_comp"].append(1)
        else: ccs_modified["dm_w_comp"].append(0)
    print("Completed Recoding of: 'dm_w_comp' ")

    for _ in df['other_endocrine']:
        ccs_modified["other_endocrine"].append(_)
    print("Completed Recoding of: 'other_endocrine' ")

    for _ in df['nutrition']:
        ccs_modified["nutrition"].append(_)
    print("Completed Recoding of: 'nutrition' ")

    for _ in df['lipid_metabo']:
        if _ >= 1:
            ccs_modified["lipid_metabo"].append(1)
        else: ccs_modified["lipid_metabo"].append(0)
    print("Completed Recoding of: 'lipid_metabo' ")

    for _ in df['gout']:
        ccs_modified["gout"].append(_)
    print("Completed Recoding of: 'gout' ")

    for _ in df['fluid_electrolyte']:
        ccs_modified["fluid_electrolyte"].append(_)
    print("Completed Recoding of: 'fluid_electrolyte' ")

    for _ in df['cyst_fibrosis']:
        ccs_modified["cyst_fibrosis"].append(_)
    print("Completed Recoding of: 'cyst_fibrosis' ")

    for _ in df['immunity']:
        if _ >= 1:
            ccs_modified["immunity"].append(1)
        else: ccs_modified["immunity"].append(0)
    print("Completed Recoding of: 'immunity' ")

    for _ in df['other_metabo']:
        ccs_modified["other_metabo"].append(_)
    print("Completed Recoding of: 'other_metabo' ")

    for _ in df['other_anemia']:
        if _ >= 1:
            ccs_modified["other_anemia"].append(1)
        else: ccs_modified["other_anemia"].append(0)
    print("Completed Recoding of: 'other_anemia' ")

    for _ in df['post_hemorr_anemia']:
        if _ >= 1:
            ccs_modified["post_hemorr_anemia"].append(1)
        else: ccs_modified["post_hemorr_anemia"].append(0)
    print("Completed Recoding of: 'post_hemorr_anemia' ")

    for _ in df['sickle_cell']:
        if _ >= 1:
            ccs_modified["sickle_cell"].append(1)
        else: ccs_modified["sickle_cell"].append(0)
    print("Completed Recoding of: 'sickle_cell' ")

    for _ in df['coag_anemia']:
        ccs_modified["coag_anemia"].append(_)
    print("Completed Recoding of: 'coag_anemia' ")

    for _ in df['wbc_disease']:
        ccs_modified["wbc_disease"].append(_)
    print("Completed Recoding of: 'wbc_disease' ")

    for _ in df['other_heme']:
        ccs_modified["other_heme"].append(_)
    print("Completed Recoding of: 'other_heme' ")

    for _ in df['meningitis_notb']:
        ccs_modified["meningitis_notb"].append(_)
    print("Completed Recoding of: 'meningitis_notb' ")

    for _ in df['encephalitis_notb']:
        ccs_modified["encephalitis_notb"].append(_)
    print("Completed Recoding of: 'encephalitis_notb' ")

    for _ in df['other_cns']:
        ccs_modified["other_cns"].append(_)
    print("Completed Recoding of: 'other_cns' ")

    for _ in df['parkinsons']:
        ccs_modified["parkinsons"].append(_)
    print("Completed Recoding of: 'parkinsons' ")

    for _ in df['mult_scler']:
        if _ >= 1:
            ccs_modified["mult_scler"].append(1)
        else: ccs_modified["mult_scler"].append(0)
    print("Completed Recoding of: 'mult_scler' ")

    for _ in df['other_hered_degen']:
        if _ >= 1:
            ccs_modified["other_hered_degen"].append(1)
        else: ccs_modified["other_hered_degen"].append(0)
    print("Completed Recoding of: 'other_hered_degen' ")

    for _ in df['paralysis']:
        if _ >= 1:
            ccs_modified["paralysis"].append(1)
        else: ccs_modified["paralysis"].append(0)
    print("Completed Recoding of: 'paralysis' ")

    for _ in df['epilepsy']:
        if _ >= 1:
            ccs_modified["epilepsy"].append(1)
        else: ccs_modified["epilepsy"].append(0)
    print("Completed Recoding of: 'epilepsy' ")

    for _ in df['headache']:
        if _ >= 1:
            ccs_modified["headache"].append(1)
        else: ccs_modified["headache"].append(0)
    print("Completed Recoding of: 'headache' ")

    for _ in df['coma']:
        ccs_modified["coma"].append(_)
    print("Completed Recoding of: 'coma' ")

    for _ in df['cataract']:
        if _ >= 1:
            ccs_modified["cataract"].append(1)
        else: ccs_modified["cataract"].append(0)
    print("Completed Recoding of: 'cataract' ")

    for _ in df['retinopathy']:
        if _ >= 1:
            ccs_modified["retinopathy"].append(1)
        else: ccs_modified["retinopathy"].append(0)
    print("Completed Recoding of: 'retinopathy' ")

    for _ in df['glaucoma']:
        if _ >= 1:
            ccs_modified["glaucoma"].append(1)
        else: ccs_modified["glaucoma"].append(0)
    print("Completed Recoding of: 'glaucoma' ")

    for _ in df['blindness']:
        ccs_modified["blindness"].append(_)
    print("Completed Recoding of: 'blindness' ")

    for _ in df['eye_inflam']:
        ccs_modified["eye_inflam"].append(_)
    print("Completed Recoding of: 'eye_inflam' ")

    for _ in df['other_eye']:
        ccs_modified["other_eye"].append(_)
    print("Completed Recoding of: 'other_eye' ")

    for _ in df['otitis_media']:
        ccs_modified["otitis_media"].append(_)
    print("Completed Recoding of: 'otitis_media' ")

    for _ in df['dizzy']:
        ccs_modified["dizzy"].append(_)
    print("Completed Recoding of: 'dizzy' ")

    for _ in df['other_ear_sense']:
        ccs_modified["other_ear_sense"].append(_)
    print("Completed Recoding of: 'other_ear_sense' ")

    for _ in df['other_ns_disorder']:
        ccs_modified["other_ns_disorder"].append(_)
    print("Completed Recoding of: 'other_ns_disorder' ")

    for _ in df['heart_valve']:
        ccs_modified["heart_valve"].append(_)
    print("Completed Recoding of: 'heart_valve' ")

    for _ in df['peri_endo_carditis']:
        ccs_modified["peri_endo_carditis"].append(_)
    print("Completed Recoding of: 'peri_endo_carditis' ")

    for _ in df['essential_htn']:
        if _ >= 1:
            ccs_modified["essential_htn"].append(1)
        else: ccs_modified["essential_htn"].append(0)
    print("Completed Recoding of: 'essential_htn' ")

    for _ in df['htn_w_comp']:
        if _ >= 1:
            ccs_modified["htn_w_comp"].append(1)
        else: ccs_modified["htn_w_comp"].append(0)
    print("Completed Recoding of: 'htn_w_comp' ")

    for _ in df['acute_mi']:
        if _ >= 1:
            ccs_modified["acute_mi"].append(1)
        else: ccs_modified["acute_mi"].append(0)
    print("Completed Recoding of: 'acute_mi' ")

    for _ in df['coronary_athero']:
        if _ >= 1:
            ccs_modified["coronary_athero"].append(1)
        else: ccs_modified["coronary_athero"].append(0)
    print("Completed Recoding of: 'coronary_athero' ")

    for _ in df['chest_pain_nos']:
        ccs_modified["chest_pain_nos"].append(_)
    print("Completed Recoding of: 'chest_pain_nos' ")

    for _ in df['pulmonary_hd']:
        ccs_modified["pulmonary_hd"].append(_)
    print("Completed Recoding of: 'pulmonary_hd' ")

    for _ in df['other_heart_disease']:
        ccs_modified["other_heart_disease"].append(_)
    print("Completed Recoding of: 'other_heart_disease' ")

    for _ in df['conduction']:
        ccs_modified["conduction"].append(_)
    print("Completed Recoding of: 'conduction' ")

    for _ in df['cardiac_dysrhythm']:
        if _ >= 1:
            ccs_modified["cardiac_dysrhythm"].append(1)
        else: ccs_modified["cardiac_dysrhythm"].append(0)
    print("Completed Recoding of: 'cardiac_dysrhythm' ")

    for _ in df['cardiac_arrest']:
        ccs_modified["cardiac_arrest"].append(_)
    print("Completed Recoding of: 'cardiac_arrest' ")

    for _ in df['chf']:
        if _ >= 1:
            ccs_modified["chf"].append(1)
        else: ccs_modified["chf"].append(0)
    print("Completed Recoding of: 'chf' ")

    for _ in df['acute_cvd']:
        if _ >= 1:
            ccs_modified["acute_cvd"].append(1)
        else: ccs_modified["acute_cvd"].append(0)
    print("Completed Recoding of: 'acute_cvd' ")

    for _ in df['occlu_cereb_artery']:
        ccs_modified["occlu_cereb_artery"].append(_)
    print("Completed Recoding of: 'occlu_cereb_artery' ")

    for _ in df['other_cvd']:
        ccs_modified["other_cvd"].append(_)
    print("Completed Recoding of: 'other_cvd' ")

    for _ in df['tran_cereb_isch']:
        if _ >= 1:
            ccs_modified["tran_cereb_isch"].append(1)
        else: ccs_modified["tran_cereb_isch"].append(0)
    print("Completed Recoding of: 'tran_cereb_isch' ")

    for _ in df['late_effect_cvd']:
        ccs_modified["late_effect_cvd"].append(_)
    print("Completed Recoding of: 'late_effect_cvd' ")

    for _ in df['pvd']:
        if _ >= 1:
            ccs_modified["pvd"].append(1)
        else: ccs_modified["pvd"].append(0)
    print("Completed Recoding of: 'pvd' ")

    for _ in df['artery_aneurysm']:
        ccs_modified["artery_aneurysm"].append(_)
    print("Completed Recoding of: 'artery_aneurysm' ")

    for _ in df['artery_embolism']:
        ccs_modified["artery_embolism"].append(_)
    print("Completed Recoding of: 'artery_embolism' ")

    for _ in df['other_circ']:
        ccs_modified["other_circ"].append(_)
    print("Completed Recoding of: 'other_circ' ")

    for _ in df['phlebitis']:
        ccs_modified["phlebitis"].append(_)
    print("Completed Recoding of: 'phlebitis' ")

    for _ in df['varicose_vein']:
        ccs_modified["varicose_vein"].append(_)
    print("Completed Recoding of: 'varicose_vein' ")

    for _ in df['hemorrhoid']:
        ccs_modified["hemorrhoid"].append(_)
    print("Completed Recoding of: 'hemorrhoid' ")

    for _ in df['other_vein_lymph']:
        ccs_modified["other_vein_lymph"].append(_)
    print("Completed Recoding of: 'other_vein_lymph' ")

    for _ in df['pneumonia']:
        ccs_modified["pneumonia"].append(_)
    print("Completed Recoding of: 'pneumonia' ")

    for _ in df['influenza']:
        ccs_modified["influenza"].append(_)
    print("Completed Recoding of: 'influenza' ")

    for _ in df['acute_tonsil']:
        ccs_modified["acute_tonsil"].append(_)
    print("Completed Recoding of: 'acute_tonsil' ")

    for _ in df['acute_bronch']:
        ccs_modified["acute_bronch"].append(_)
    print("Completed Recoding of: 'acute_bronch' ")

    for _ in df['upper_resp_infec']:
        ccs_modified["upper_resp_infec"].append(_)
    print("Completed Recoding of: 'upper_resp_infec' ")

    for _ in df['copd']:
        if _ >= 1:
            ccs_modified["copd"].append(1)
        else: ccs_modified["copd"].append(0)
    print("Completed Recoding of: 'copd' ")

    for _ in df['asthma']:
        if _ >= 1:
            ccs_modified["asthma"].append(1)
        else: ccs_modified["asthma"].append(0)
    print("Completed Recoding of: 'asthma' ")

    for _ in df['asp_pneumonitis']:
        ccs_modified["asp_pneumonitis"].append(_)
    print("Completed Recoding of: 'asp_pneumonitis' ")

    for _ in df['pneumothorax']:
        ccs_modified["pneumothorax"].append(_)
    print("Completed Recoding of: 'pneumothorax' ")

    for _ in df['resp_failure']:
        ccs_modified["resp_failure"].append(_)
    print("Completed Recoding of: 'resp_failure' ")

    for _ in df['lung_disease']:
        ccs_modified["lung_disease"].append(_)
    print("Completed Recoding of: 'lung_disease' ")

    for _ in df['other_low_resp']:
        ccs_modified["other_low_resp"].append(_)
    print("Completed Recoding of: 'other_low_resp' ")

    for _ in df['other_up_resp']:
        ccs_modified["other_up_resp"].append(_)
    print("Completed Recoding of: 'other_up_resp' ")

    for _ in df['intestinal_infec']:
        ccs_modified["intestinal_infec"].append(_)
    print("Completed Recoding of: 'intestinal_infec' ")

    for _ in df['teeth_jaw']:
        ccs_modified["teeth_jaw"].append(_)
    print("Completed Recoding of: 'teeth_jaw' ")

    for _ in df['mouth_disease']:
        ccs_modified["mouth_disease"].append(_)
    print("Completed Recoding of: 'mouth_disease' ")

    for _ in df['esophagus']:
        ccs_modified["esophagus"].append(_)
    print("Completed Recoding of: 'esophagus' ")

    for _ in df['gastro_ulcer']:
        ccs_modified["gastro_ulcer"].append(_)
    print("Completed Recoding of: 'gastro_ulcer' ")

    for _ in df['gastritis']:
        ccs_modified["gastritis"].append(_)
    print("Completed Recoding of: 'gastritis' ")

    for _ in df['other_stomach_duo']:
        ccs_modified["other_stomach_duo"].append(_)
    print("Completed Recoding of: 'other_stomach_duo' ")

    for _ in df['appendicitis']:
        ccs_modified["appendicitis"].append(_)
    print("Completed Recoding of: 'appendicitis' ")

    for _ in df['hernia_abd']:
        ccs_modified["hernia_abd"].append(_)
    print("Completed Recoding of: 'hernia_abd' ")

    for _ in df['regional_enteriritis']:
        ccs_modified["regional_enteriritis"].append(_)
    print("Completed Recoding of: 'regional_enteriritis' ")

    for _ in df['intestinal_obstruct']:
        ccs_modified["intestinal_obstruct"].append(_)
    print("Completed Recoding of: 'intestinal_obstruct' ")

    for _ in df['diverticulitis']:
        ccs_modified["diverticulitis"].append(_)
    print("Completed Recoding of: 'diverticulitis' ")

    for _ in df['anal_condition']:
        ccs_modified["anal_condition"].append(_)
    print("Completed Recoding of: 'anal_condition' ")

    for _ in df['peritonitis']:
        ccs_modified["peritonitis"].append(_)
    print("Completed Recoding of: 'peritonitis' ")

    for _ in df['biliary_tract']:
        ccs_modified["biliary_tract"].append(_)
    print("Completed Recoding of: 'biliary_tract' ")

    for _ in df['other_liver']:
        if _ >= 1:
            ccs_modified["other_liver"].append(1)
        else: ccs_modified["other_liver"].append(0)
    print("Completed Recoding of: 'other_liver' ")

    for _ in df['pancreatic']:
        ccs_modified["pancreatic"].append(_)
    print("Completed Recoding of: 'pancreatic' ")

    for _ in df['gastro_hemorrhage']:
        ccs_modified["gastro_hemorrhage"].append(_)
    print("Completed Recoding of: 'gastro_hemorrhage' ")

    for _ in df['noninfec_gastro']:
        ccs_modified["noninfec_gastro"].append(_)
    print("Completed Recoding of: 'noninfec_gastro' ")

    for _ in df['other_gastro']:
        ccs_modified["other_gastro"].append(_)
    print("Completed Recoding of: 'other_gastro' ")

    for _ in df['nephritis']:
        if _ >= 1:
            ccs_modified["nephritis"].append(1)
        else: ccs_modified["nephritis"].append(0)
    print("Completed Recoding of: 'nephritis' ")

    for _ in df['acute_renal_fail']:
        if _ >= 1:
            ccs_modified["acute_renal_fail"].append(1)
        else: ccs_modified["acute_renal_fail"].append(0)
    print("Completed Recoding of: 'acute_renal_fail' ")

    for _ in df['ckd']:
        if _ >= 1:
            ccs_modified["ckd"].append(1)
        else: ccs_modified["ckd"].append(0)
    print("Completed Recoding of: 'ckd' ")

    for _ in df['uti']:
        ccs_modified["uti"].append(_)
    print("Completed Recoding of: 'uti' ")

    for _ in df['calculus_urinary']:
        ccs_modified["calculus_urinary"].append(_)
    print("Completed Recoding of: 'calculus_urinary' ")

    for _ in df['other_kidney']:
        if _ >= 1:
            ccs_modified["other_kidney"].append(1)
        else: ccs_modified["other_kidney"].append(0)
    print("Completed Recoding of: 'other_kidney' ")

    for _ in df['other_bladder']:
        ccs_modified["other_bladder"].append(_)
    print("Completed Recoding of: 'other_bladder' ")

    for _ in df['genitourinary_symp']:
        ccs_modified["genitourinary_symp"].append(_)
    print("Completed Recoding of: 'genitourinary_symp' ")

    for _ in df['prostate_hyp']:
        if _ >= 1:
            ccs_modified["prostate_hyp"].append(1)
        else: ccs_modified["prostate_hyp"].append(0)
    print("Completed Recoding of: 'prostate_hyp' ")

    for _ in df['male_genital_inflam']:
        ccs_modified["male_genital_inflam"].append(_)
    print("Completed Recoding of: 'male_genital_inflam' ")

    for _ in df['other_male_genital']:
        ccs_modified["other_male_genital"].append(_)
    print("Completed Recoding of: 'other_male_genital' ")

    for _ in df['nonmalig_breast']:
        ccs_modified["nonmalig_breast"].append(_)
    print("Completed Recoding of: 'nonmalig_breast' ")

    for _ in df['inflam_fem_pelvic']:
        ccs_modified["inflam_fem_pelvic"].append(_)
    print("Completed Recoding of: 'inflam_fem_pelvic' ")

    for _ in df['endometriosis']:
        ccs_modified["endometriosis"].append(_)
    print("Completed Recoding of: 'endometriosis' ")

    for _ in df['prolapse_fem_gen']:
        ccs_modified["prolapse_fem_gen"].append(_)
    print("Completed Recoding of: 'prolapse_fem_gen' ")

    for _ in df['menstrual']:
        ccs_modified["menstrual"].append(_)
    print("Completed Recoding of: 'menstrual' ")

    for _ in df['ovarian_cyst']:
        ccs_modified["ovarian_cyst"].append(_)
    print("Completed Recoding of: 'ovarian_cyst' ")

    for _ in df['menopausal']:
        ccs_modified["menopausal"].append(_)
    print("Completed Recoding of: 'menopausal' ")

    for _ in df['fem_infert']:
        ccs_modified["fem_infert"].append(_)
    print("Completed Recoding of: 'fem_infert' ")

    for _ in df['other_fem_genital']:
        ccs_modified["other_fem_genital"].append(_)
    print("Completed Recoding of: 'other_fem_genital' ")

    for _ in df['contraceptive_mgmt']:
        ccs_modified["contraceptive_mgmt"].append(_)
    print("Completed Recoding of: 'contraceptive_mgmt' ")

    for _ in df['spont_abortion']:
        ccs_modified["spont_abortion"].append(_)
    print("Completed Recoding of: 'spont_abortion' ")

    for _ in df['induce_abortion']:
        ccs_modified["induce_abortion"].append(_)
    print("Completed Recoding of: 'induce_abortion' ")

    for _ in df['postabort_comp']:
        ccs_modified["postabort_comp"].append(_)
    print("Completed Recoding of: 'postabort_comp' ")

    for _ in df['ectopic_preg']:
        ccs_modified["ectopic_preg"].append(_)
    print("Completed Recoding of: 'ectopic_preg' ")

    for _ in df['other_comp_preg']:
        ccs_modified["other_comp_preg"].append(_)
    print("Completed Recoding of: 'other_comp_preg' ")

    for _ in df['hemorrhage_preg']:
        ccs_modified["hemorrhage_preg"].append(_)
    print("Completed Recoding of: 'hemorrhage_preg' ")

    for _ in df['htn_comp_preg']:
        ccs_modified["htn_comp_preg"].append(_)
    print("Completed Recoding of: 'htn_comp_preg' ")

    for _ in df['early_labor']:
        ccs_modified["early_labor"].append(_)
    print("Completed Recoding of: 'early_labor' ")

    for _ in df['prolong_preg']:
        ccs_modified["prolong_preg"].append(_)
    print("Completed Recoding of: 'prolong_preg' ")

    for _ in df['dm_comp_preg']:
        ccs_modified["dm_comp_preg"].append(_)
    print("Completed Recoding of: 'dm_comp_preg' ")

    for _ in df['malposition']:
        ccs_modified["malposition"].append(_)
    print("Completed Recoding of: 'malposition' ")

    for _ in df['fetopelvic_disrupt']:
        ccs_modified["fetopelvic_disrupt"].append(_)
    print("Completed Recoding of: 'fetopelvic_disrupt' ")

    for _ in df['prev_c_sect']:
        ccs_modified["prev_c_sect"].append(_)
    print("Completed Recoding of: 'prev_c_sect' ")

    for _ in df['fetal_distress']:
        ccs_modified["fetal_distress"].append(_)
    print("Completed Recoding of: 'fetal_distress' ")

    for _ in df['polyhydramnios']:
        ccs_modified["polyhydramnios"].append(_)
    print("Completed Recoding of: 'polyhydramnios' ")

    for _ in df['umbilical_comp']:
        ccs_modified["umbilical_comp"].append(_)
    print("Completed Recoding of: 'umbilical_comp' ")

    for _ in df['ob_trauma']:
        ccs_modified["ob_trauma"].append(_)
    print("Completed Recoding of: 'ob_trauma' ")

    for _ in df['forceps_deliv']:
        ccs_modified["forceps_deliv"].append(_)
    print("Completed Recoding of: 'forceps_deliv' ")

    for _ in df['other_comp_birth']:
        ccs_modified["other_comp_birth"].append(_)
    print("Completed Recoding of: 'other_comp_birth' ")

    for _ in df['other_preg_deliv']:
        ccs_modified["other_preg_deliv"].append(_)
    print("Completed Recoding of: 'other_preg_deliv' ")

    for _ in df['skin_tissue_infec']:
        ccs_modified["skin_tissue_infec"].append(_)
    print("Completed Recoding of: 'skin_tissue_infec' ")

    for _ in df['other_skin_inflam']:
        ccs_modified["other_skin_inflam"].append(_)
    print("Completed Recoding of: 'other_skin_inflam' ")

    for _ in df['chronic_skin_ulcer']:
        if _ >= 1:
            ccs_modified["chronic_skin_ulcer"].append(1)
        else: ccs_modified["chronic_skin_ulcer"].append(0)
    print("Completed Recoding of: 'chronic_skin_ulcer' ")

    for _ in df['other_skin']:
        ccs_modified["other_skin"].append(_)
    print("Completed Recoding of: 'other_skin' ")

    for _ in df['infec_arthritis']:
        ccs_modified["infec_arthritis"].append(_)
    print("Completed Recoding of: 'infec_arthritis' ")

    for _ in df['rheum_arth']:
        if _ >= 1:
            ccs_modified["rheum_arth"].append(1)
        else: ccs_modified["rheum_arth"].append(0)
    print("Completed Recoding of: 'rheum_arth' ")

    for _ in df['osteo_arth']:
        if _ >= 1:
            ccs_modified["osteo_arth"].append(1)
        else: ccs_modified["osteo_arth"].append(0)
    print("Completed Recoding of: 'osteo_arth' ")

    for _ in df['other_joint']:
        ccs_modified["other_joint"].append(_)
    print("Completed Recoding of: 'other_joint' ")

    for _ in df['spondylosis']:
        if _ >= 1:
            ccs_modified["spondylosis"].append(1)
        else: ccs_modified["spondylosis"].append(0)
    print("Completed Recoding of: 'spondylosis' ")

    for _ in df['osteoporosis']:
        if _ >= 1:
            ccs_modified["osteoporosis"].append(1)
        else: ccs_modified["osteoporosis"].append(0)
    print("Completed Recoding of: 'osteoporosis' ")

    for _ in df['pathological_fract']:
        if _ >= 1:
            ccs_modified["pathological_fract"].append(1)
        else: ccs_modified["pathological_fract"].append(0)
    print("Completed Recoding of: 'pathological_fract' ")

    for _ in df['acq_foot_deform']:
        ccs_modified["acq_foot_deform"].append(_)
    print("Completed Recoding of: 'acq_foot_deform' ")

    for _ in df['other_acq_deform']:
        ccs_modified["other_acq_deform"].append(_)
    print("Completed Recoding of: 'other_acq_deform' ")

    for _ in df['systemic_lupus']:
        ccs_modified["systemic_lupus"].append(_)
    print("Completed Recoding of: 'systemic_lupus' ")

    for _ in df['other_connective']:
        ccs_modified["other_connective"].append(_)
    print("Completed Recoding of: 'other_connective' ")

    for _ in df['other_bone_disease']:
        ccs_modified["other_bone_disease"].append(_)
    print("Completed Recoding of: 'other_bone_disease' ")

    for _ in df['cardiac_congen_anom']:
        ccs_modified["cardiac_congen_anom"].append(_)
    print("Completed Recoding of: 'cardiac_congen_anom' ")

    for _ in df['digest_congen_anom']:
        ccs_modified["digest_congen_anom"].append(_)
    print("Completed Recoding of: 'digest_congen_anom' ")

    for _ in df['genito_congen_anom']:
        ccs_modified["genito_congen_anom"].append(_)
    print("Completed Recoding of: 'genito_congen_anom' ")

    for _ in df['ns_congen_anom']:
        if _ >= 1:
            ccs_modified["ns_congen_anom"].append(1)
        else: ccs_modified["ns_congen_anom"].append(0)
    print("Completed Recoding of: 'ns_congen_anom' ")

    for _ in df['other_congen_anom']:
        if _ >= 1:
            ccs_modified["other_congen_anom"].append(1)
        else: ccs_modified["other_congen_anom"].append(0)
    print("Completed Recoding of: 'other_congen_anom' ")

    for _ in df['liveborn']:
        ccs_modified["liveborn"].append(_)
    print("Completed Recoding of: 'liveborn' ")

    for _ in df['short_gest']:
        ccs_modified["short_gest"].append(_)
    print("Completed Recoding of: 'short_gest' ")

    for _ in df['intrauter_hypoxia']:
        ccs_modified["intrauter_hypoxia"].append(_)
    print("Completed Recoding of: 'intrauter_hypoxia' ")

    for _ in df['resp_distress_synd']:
        ccs_modified["resp_distress_synd"].append(_)
    print("Completed Recoding of: 'resp_distress_synd' ")

    for _ in df['hemolytic_jaundice']:
        ccs_modified["hemolytic_jaundice"].append(_)
    print("Completed Recoding of: 'hemolytic_jaundice' ")

    for _ in df['birth_trauma']:
        ccs_modified["birth_trauma"].append(_)
    print("Completed Recoding of: 'birth_trauma' ")

    for _ in df['other_perinatal']:
        ccs_modified["other_perinatal"].append(_)
    print("Completed Recoding of: 'other_perinatal' ")

    for _ in df['joint_trauma']:
        ccs_modified["joint_trauma"].append(_)
    print("Completed Recoding of: 'joint_trauma' ")

    for _ in df['fract_femur_neck']:
        if _ >= 1:
            ccs_modified["fract_femur_neck"].append(1)
        else: ccs_modified["fract_femur_neck"].append(0)
    print("Completed Recoding of: 'fract_femur_neck' ")

    for _ in df['spinal_cord']:
        if _ >= 1:
            ccs_modified["spinal_cord"].append(1)
        else: ccs_modified["spinal_cord"].append(0)
    print("Completed Recoding of: 'spinal_cord' ")

    for _ in df['skull_face_fract']:
        ccs_modified["skull_face_fract"].append(_)
    print("Completed Recoding of: 'skull_face_fract' ")

    for _ in df['upper_limb_fract']:
        ccs_modified["upper_limb_fract"].append(_)
    print("Completed Recoding of: 'upper_limb_fract' ")

    for _ in df['lower_limb_fract']:
        ccs_modified["lower_limb_fract"].append(_)
    print("Completed Recoding of: 'lower_limb_fract' ")

    for _ in df['other_fract']:
        if _ >= 1:
            ccs_modified["other_fract"].append(1)
        else: ccs_modified["other_fract"].append(0)
    print("Completed Recoding of: 'other_fract' ")

    for _ in df['sprain_strain']:
        ccs_modified["sprain_strain"].append(_)
    print("Completed Recoding of: 'sprain_strain' ")

    for _ in df['intracranial']:
        ccs_modified["intracranial"].append(_)
    print("Completed Recoding of: 'intracranial' ")

    for _ in df['crush_injury']:
        ccs_modified["crush_injury"].append(_)
    print("Completed Recoding of: 'crush_injury' ")

    for _ in df['open_wound_head']:
        ccs_modified["open_wound_head"].append(_)
    print("Completed Recoding of: 'open_wound_head' ")

    for _ in df['open_wound_extr']:
        ccs_modified["open_wound_extr"].append(_)
    print("Completed Recoding of: 'open_wound_extr' ")

    for _ in df['comp_of_device']:
        ccs_modified["comp_of_device"].append(_)
    print("Completed Recoding of: 'comp_of_device' ")

    for _ in df['comp_surg_proc']:
        ccs_modified["comp_surg_proc"].append(_)
    print("Completed Recoding of: 'comp_surg_proc' ")

    for _ in df['superficial_inj']:
        ccs_modified["superficial_inj"].append(_)
    print("Completed Recoding of: 'superficial_inj' ")

    for _ in df['burns']:
        ccs_modified["burns"].append(_)
    print("Completed Recoding of: 'burns' ")

    for _ in df['poison_psycho']:
        ccs_modified["poison_psycho"].append(_)
    print("Completed Recoding of: 'poison_psycho' ")

    for _ in df['poison_other_med']:
        ccs_modified["poison_other_med"].append(_)
    print("Completed Recoding of: 'poison_other_med' ")

    for _ in df['poison_nonmed']:
        ccs_modified["poison_nonmed"].append(_)
    print("Completed Recoding of: 'poison_nonmed' ")

    for _ in df['other_ext_injury']:
        ccs_modified["other_ext_injury"].append(_)
    print("Completed Recoding of: 'other_ext_injury' ")

    for _ in df['syncope']:
        ccs_modified["syncope"].append(_)
    print("Completed Recoding of: 'syncope' ")

    for _ in df['fever_unknown']:
        ccs_modified["fever_unknown"].append(_)
    print("Completed Recoding of: 'fever_unknown' ")

    for _ in df['lymphadenitis']:
        ccs_modified["lymphadenitis"].append(_)
    print("Completed Recoding of: 'lymphadenitis' ")

    for _ in df['gangrene']:
        ccs_modified["gangrene"].append(_)
    print("Completed Recoding of: 'gangrene' ")

    for _ in df['shock']:
        ccs_modified["shock"].append(_)
    print("Completed Recoding of: 'shock' ")

    for _ in df['naus_vom']:
        ccs_modified["naus_vom"].append(_)
    print("Completed Recoding of: 'naus_vom' ")

    for _ in df['abdominal_pain']:
        ccs_modified["abdominal_pain"].append(_)
    print("Completed Recoding of: 'abdominal_pain' ")

    for _ in df['malaise_fatigue']:
        if _ >= 1:
            ccs_modified["malaise_fatigue"].append(1)
        else: ccs_modified["malaise_fatigue"].append(0)
    print("Completed Recoding of: 'malaise_fatigue' ")

    for _ in df['allergy']:
        ccs_modified["allergy"].append(_)
    print("Completed Recoding of: 'allergy' ")

    for _ in df['rehab_care']:
        ccs_modified["rehab_care"].append(_)
    print("Completed Recoding of: 'rehab_care' ")

    for _ in df['admin_admiss']:
        ccs_modified["admin_admiss"].append(_)
    print("Completed Recoding of: 'admin_admiss' ")

    for _ in df['medical_eval']:
        ccs_modified["medical_eval"].append(_)
    print("Completed Recoding of: 'medical_eval' ")

    for _ in df['other_aftercare']:
        ccs_modified["other_aftercare"].append(_)
    print("Completed Recoding of: 'other_aftercare' ")

    for _ in df['other_screen']:
        ccs_modified["other_screen"].append(_)
    print("Completed Recoding of: 'other_screen' ")

    for _ in df['residual_codes']:
        ccs_modified["residual_codes"].append(_)
    print("Completed Recoding of: 'residual_codes' ")

    for _ in df['adjustment']:
        ccs_modified["adjustment"].append(_)
    print("Completed Recoding of: 'adjustment' ")

    for _ in df['anxiety']:
        ccs_modified["anxiety"].append(_)
        # if _ >= 1:
        #     ccs_modified["anxiety"].append(1)
        # else: ccs_modified["anxiety"].append(0)
    print("Completed Recoding of: 'anxiety' ")

    for _ in df['adhd']:
        if _ >= 1:
            ccs_modified["adhd"].append(1)
        else: ccs_modified["adhd"].append(0)
    print("Completed Recoding of: 'adhd' ")

    for _ in df['dementia']:
        if _ >= 1:
            ccs_modified["dementia"].append(1)
        else: ccs_modified["dementia"].append(0)
    print("Completed Recoding of: 'dementia' ")

    for _ in df['develop_dis']:
        if _ >= 1:
            ccs_modified["develop_dis"].append(1)
        else: ccs_modified["develop_dis"].append(0)
    print("Completed Recoding of: 'develop_dis' ")

    for _ in df['child_disorder']:
        if _ >= 1:
            ccs_modified["child_disorder"].append(1)
        else: ccs_modified["child_disorder"].append(0)
    print("Completed Recoding of: 'child_disorder' ")

    for _ in df['impule_control']:
        ccs_modified["impule_control"].append(_)
    print("Completed Recoding of: 'impule_control' ")

    for _ in df['mood']:
        ccs_modified["mood"].append(_)
        # if _ >= 1:
        #     ccs_modified["mood"].append(1)
        # else: ccs_modified["mood"].append(0)
    print("Completed Recoding of: 'mood' ")

    for _ in df['personality']:
        ccs_modified["personality"].append(_)
        # if _ >= 1:
        #     ccs_modified["personality"].append(1)
        # else: ccs_modified["personality"].append(0)
    print("Completed Recoding of: 'personality' ")

    for _ in df['schizo']:
        ccs_modified["schizo"].append(_)
        # if _ >= 1:
        #     ccs_modified["schizo"].append(1)
        # else: ccs_modified["schizo"].append(0)
    print("Completed Recoding of: 'schizo' ")

    for _ in df['alcohol']:
        ccs_modified["alcohol"].append(_)
    print("Completed Recoding of: 'alcohol' ")

    for _ in df['substance']:
        ccs_modified["substance"].append(_)
    print("Completed Recoding of: 'substance' ")

    for _ in df['suicide']:
        ccs_modified["suicide"].append(_)
    print("Completed Recoding of: 'suicide' ")

    for _ in df['mental_screen']:
        ccs_modified["mental_screen"].append(_)
    print("Completed Recoding of: 'mental_screen' ")

    for _ in df['misc_mental']:
        ccs_modified["misc_mental"].append(_)
    print("Completed Recoding of: 'misc_mental' ")

    for _ in df['e_cut_pierce']:
        ccs_modified["e_cut_pierce"].append(_)
    print("Completed Recoding of: 'e_cut_pierce' ")

    for _ in df['e_drown']:
        ccs_modified["e_drown"].append(_)
    print("Completed Recoding of: 'e_drown' ")

    for _ in df['e_fall']:
        ccs_modified["e_fall"].append(_)
    print("Completed Recoding of: 'e_fall' ")

    for _ in df['e_fire']:
        ccs_modified["e_fire"].append(_)
    print("Completed Recoding of: 'e_fire' ")

    for _ in df['e_firearm']:
        ccs_modified["e_firearm"].append(_)
    print("Completed Recoding of: 'e_firearm' ")

    for _ in df['e_machine']:
        ccs_modified["e_machine"].append(_)
    print("Completed Recoding of: 'e_machine' ")

    for _ in df['e_mvt']:
        ccs_modified["e_mvt"].append(_)
    print("Completed Recoding of: 'e_mvt' ")

    for _ in df['e_cyclist']:
        ccs_modified["e_cyclist"].append(_)
    print("Completed Recoding of: 'e_cyclist' ")

    for _ in df['e_pedestrian']:
        ccs_modified["e_pedestrian"].append(_)
    print("Completed Recoding of: 'e_pedestrian' ")

    for _ in df['e_transport']:
        ccs_modified["e_transport"].append(_)
    print("Completed Recoding of: 'e_transport' ")

    for _ in df['e_natural']:
        ccs_modified["e_natural"].append(_)
    print("Completed Recoding of: 'e_natural' ")

    for _ in df['e_overexert']:
        ccs_modified["e_overexert"].append(_)
    print("Completed Recoding of: 'e_overexert' ")

    for _ in df['e_poison']:
        ccs_modified["e_poison"].append(_)
    print("Completed Recoding of: 'e_poison' ")

    for _ in df['e_struckby']:
        ccs_modified["e_struckby"].append(_)
    print("Completed Recoding of: 'e_struckby' ")

    for _ in df['e_suffocate']:
        ccs_modified["e_suffocate"].append(_)
    print("Completed Recoding of: 'e_suffocate' ")

    for _ in df['e_ae_med_care']:
        ccs_modified["e_ae_med_care"].append(_)
    print("Completed Recoding of: 'e_ae_med_care' ")

    for _ in df['e_ae_med_drug']:
        ccs_modified["e_ae_med_drug"].append(_)
    print("Completed Recoding of: 'e_ae_med_drug' ")

    for _ in df['e_other_class']:
        ccs_modified["e_other_class"].append(_)
    print("Completed Recoding of: 'e_other_class' ")

    for _ in df['e_other_nec']:
        ccs_modified["e_other_nec"].append(_)
    print("Completed Recoding of: 'e_other_nec' ")

    for _ in df['e_unspecified']:
        ccs_modified["e_unspecified"].append(_)
    print("Completed Recoding of: 'e_unspecified' ")

    for _ in df['e_place']:
        ccs_modified["e_place"].append(_)
    print("Completed Recoding of: 'e_place' ")


    df2 = pd.DataFrame(ccs_modified)

    df2 = df2.merge(visits, on='person_id', how='left')

    return df2


# def add_location(df):
#     loc = pd.read_csv(PATH+'cdrn_dedup_loc_all.csv', encoding='utf-8')
#
#     df = pd.merge(df, loc, how='inner', on='person_id')
#     print("cdrn_ccs_modified_loc unique patients: ", df.person_id.nunique())
#
#     return df


def hosp_label(df):
    cdrn_psych_hosp_bef_18 = pd.read_csv(
        PATH+'cdrn_psych_hosp_bef_18.csv', encoding='utf-8',
        names=['person_id', 'inpt_hosp_date', 'visit_occurrence_id', 'visit_concept_id', 'condition_type_concept_id'], delimiter="\t")

    person_list2 = []
    for _ in df['person_id']:
        person_list2.append(_)

    psych_hosp_list_final = []
    for _ in df['person_id']:
        if _ in (list(cdrn_psych_hosp_bef_18['person_id'])):
            psych_hosp_list_final.append(1)
        else: psych_hosp_list_final.append(0)

    person_psych = pd.DataFrame({"person_id":person_list2,"psych_hosp":psych_hosp_list_final})
    df = pd.merge(df,person_psych, how='left', on='person_id')

    return df


def no_dx_hist(df):
    # Reset Index
    df = df.reset_index(drop=True)

    dx_cols = list(df.columns[1:284])
    df['no_dx'] = df[dx_cols].sum(axis=1)

    no_dx_flag = []
    for _ in df['no_dx']:
        if _ == 0:
            no_dx_flag.append(1)
        else: no_dx_flag.append(0)
    no_dx_flag_ = pd.DataFrame({"no_dx_flag":no_dx_flag})

    df = pd.merge(df,no_dx_flag_, left_index=True, right_index=True)

    df2 = df.merge(case_control, on='person_id', how='inner')

    pd.DataFrame(df2).to_csv(PATH+'cdrn_ccs_modified_label2.csv',encoding='utf-8')
    return df2

# def nta_data(df):
#     nta = pd.read_csv(
#         '/Users/jdeferio/Documents/Work/Cornell/Social Behavior R01/Projects/SDH/NYC Neighborhood '
#         'Data/nyc_nta_sdoh.csv', encoding='utf-8')
#     df = pd.merge(df, nta, how='left', on='nta_code')
#
#     df2 = df.merge(case_control, on='person_id', how='inner')
#
#     pd.DataFrame(df2).to_csv(PATH+'cdrn_ccs_modified_nta.csv',encoding='utf-8')
#
#     return df2


cdrn_ccs_grouped = ccs_aggregate(case_control)

print (cdrn_ccs_grouped.head(5))
print ("------------")
#sys.exit()

cdrn_ccs_modified = dx_convert(cdrn_ccs_grouped)

# cdrn_ccs_modified_loc = add_location(cdrn_ccs_modified)

cdrn_ccs_label = hosp_label(cdrn_ccs_modified)

cdrn_ccs_label_ = no_dx_hist(cdrn_ccs_label)

# cdrn_ccs_modified_nta = nta_data(cdrn_ccs_label_)
