import sys 
import os.path
from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm 
from const import URLs
from shutil import unpack_archive
from glob import glob
import json
from statistics import median 
import pickle
import json
from statistics import median 
import pickle

def download_data(url:str, dest, return_fn:str=False):
    #Modified code from: https://towardsdatascience.com/how-to-download-files-using-python-part-2-19b95be4cdb5
    
    fname = url.split('/')[-1].split('?')[0]
    
    Path(dest).mkdir(parents=True, exist_ok=True)
    fname = f'{dest}/{fname}'
        
    r = requests.get(url, stream=True, allow_redirects=True)
    total_size = int(r.headers.get('content-length'))
    initial_pos = 0
    with open(fname,'wb') as f: 
        with tqdm(total=total_size, unit='B', unit_scale=True, desc='Downloading',initial=initial_pos) as pbar:
            for ch in r.iter_content(chunk_size=1024):                          
                if ch:
                    f.write(ch) 
                    pbar.update(len(ch))
                    
    if ".tar" in fname:
        print("Extracting" + fname)
        unpack_archive(fname, str(dest))
        os.remove(fname)
    
    if return_fn: return fname

def get_img_dir(pickle_fn='../src/path_to_raw_data.pkl'):
    """Get the path for the unzipped and preprocessed images.
        
    If the argument `pickle_fn` isn't passed in, the default path is used.

    Parameters
    ----------
    pickle_fn : (str, PosixPath), optional
        The file path for the pickle object containing image path
    
    Returns
    -------
    (PosixPath, NoneType)
        Image path 
    """
    path = None
    if os.path.getsize(pickle_fn) > 0:      
        with open(pickle_fn, 'rb') as input_file:
            path = pickle.load(input_file)
    
    return path

def _get_evaluation_data(json_data, index):
    """Get the measurment value for a specific LIDC-IDRI evaluation concept.

    Parameters
    ----------
    json_data : dict 
        JSON data that describes the characterizations and measurements for each of the annotations
    index : int
        The index for a specific evaulation concept in the json file
    
    Returns
    -------
    str
        Measurement value for a specific LIDC-IDRI evaluation concept.
    """
    
    return json_data['Measurements'][-1]['qualitativeEvaluations'][index]['conceptValue']['CodeValue']

def _add_patient_level_diagnosis(df,diagnois_data_path): 
    """Add patient level diagnosis if biopsy is done. 
    
    Parameters
    ----------
    df : DataFrame
        Information about all nodules 
    diagnois_data_path : str
        The file path for the diagnosis data
    
    Returns
    -------
    DataFrame
        Added patient level diagnosis column to the dataframe. 
    """
    
    diagnosis = pd.read_excel(diagnois_data_path)
    diagnosis_df = diagnosis.iloc[:, [0,1]]
    diagnosis_df.columns = ['subject_id', 'patient_level_diagnosis']
    
    diagnosis_df = diagnosis_df.loc[diagnosis_df.patient_level_diagnosis != 0].reset_index(drop=True)
    binary_patient_level_diagnosis = [0 if diagnosis == 1 else 1 for diagnosis in diagnosis_df.patient_level_diagnosis]
    
    diagnosis_df['binary_patient_level_diagnosis'] = binary_patient_level_diagnosis
    
    for index, subject_data in diagnosis_df.iterrows(): 
        idxs = df.loc[df.subject_id == subject_data.subject_id].index
        df.loc[idxs, 'patient_level_diagnosis'] = subject_data.patient_level_diagnosis
        df.loc[idxs, 'binary_patient_level_diagnosis'] = subject_data.binary_patient_level_diagnosis
    
    return df    

def _create_lidc_idri_dataframe(metadata_path): 
    """Create a dataframe with information from the downloaded metadata and JSON files. 
    
    Parameters
    ----------
    metadata_path : str
        The file path for metadata 
    
    Returns
    -------
    DataFrame
        Information about the nodules 
    """
    
    print('Preprosessing LIDC-IDRI metadata')
    DATA_DIR = get_img_dir()
    metadata = pd.read_csv(metadata_path)
    
    
    data_list = sorted(glob(str(DATA_DIR/'LIDC_conversion5'/'*'/'*'/'*')))
    
    sphericity_dict = {'RID5811':1, '002': 2, 'RID5800':3, '004':4, 'RID5799':5}
    calcification_dict = {'RID35453':1, '302': 2, 'RID5741':3, '304':4, 'RID5827':5, 'RID28473':6}
    margin_dict = {'RID5709':1, '002': 2, '003':3, '004':4, 'RID5707':5}
    texture_dict = {'RID50153':1, '002': 2, 'RID50152':3, '004':4, 'RID50151':5}
    internal_structure_dict = {'C12471':'Soft tissue', 'C25278': 'Fluid', 'C12472':'Fat', 'C73434':'Air'}
    
    columns = ['path', 'subject_id', 'nodule','study_description','scan_session', 'study_date', 'modality','scan_description', 'manufacturer', 'manufacturer_model', 'sofware_version',
           'radiologists_lobular_score','radiologists_spiculation_score','radiologists_subtlety_score', 'radiologists_internal_structure_def', 'radiologists_calcification_def',
           'radiologists_sphericity_score', 'radiologists_margin_score', 'radiologists_texture_def', 'radiologists_malignancy_score','median_malignancy_score']
    
    df = pd.DataFrame(columns=columns)
    
    MAX_NR_NODULES = 30
    subjects_session_numb_dict = {}

    for subject_path in data_list: 
    
        series_uid = subject_path.split('/')[-1]
        subject_metadata = metadata.loc[metadata['Series UID'] == series_uid].values[0]
    
        subject_id = subject_metadata[1]
        if subject_id in subjects_session_numb_dict: subjects_session_numb_dict[subject_id]+=1
        else: subjects_session_numb_dict[subject_id]=1
    
        for i in range(MAX_NR_NODULES):
            nodule_nr = i+1
            nodule_score_files = sorted(glob(subject_path+fr'/Nodule {nodule_nr} -*measurements.json')) #r space counts in the glob search 
            if not nodule_score_files: break;
        
            radiologists_malignancy_score = []
            radiologists_lobular_score = []
            radiologists_spiculation_score = []
            radiologists_subtlety_score = []
            radiologists_internal_structure_def = []
            radiologists_calcification_def = []
            radiologists_sphericity_score = []
            radiologists_margin_score = []
            radiologists_texture_def = []
        
            for score_file in nodule_score_files: 
                with open(score_file) as json_file:
                    data = json.load(json_file)
                
                    if len(data['Measurements'][-1]['qualitativeEvaluations']) == 9: 
                    
                        subtlety_score = _get_evaluation_data(data, 0)
                        internal_structure_def = internal_structure_dict[_get_evaluation_data(data, 1)]
                        calcification_def = calcification_dict[_get_evaluation_data(data, 2)]
                        sphericity_score = sphericity_dict[_get_evaluation_data(data, 3)]
                        margin_score = margin_dict[_get_evaluation_data(data, 4)]
                        lobular_score = _get_evaluation_data(data, 5)
                        spiculation_score = _get_evaluation_data(data, 6)
                        texture_def = texture_dict[_get_evaluation_data(data, 7)]
                        malignancy_score = _get_evaluation_data(data, 8) 

                
                        radiologists_malignancy_score.append(int(malignancy_score)%10)
                        radiologists_lobular_score.append(int(lobular_score)%10)
                        radiologists_spiculation_score.append(int(spiculation_score)%10)
                        radiologists_subtlety_score.append(int(subtlety_score)%10)
                        radiologists_calcification_def.append(calcification_def)
                        radiologists_sphericity_score.append(sphericity_score)
                        radiologists_margin_score.append(margin_score)
                        radiologists_texture_def.append(texture_def)
                        radiologists_internal_structure_def.append(internal_structure_def)
                
            if radiologists_malignancy_score: 
            
                study_path = '/'.join((subject_path.split('/')[-3:]))
                study_path = study_path + f'/CT-Nodule {i+1}-union-dilated-3-masked-cropped.nrrd'
        
                df = df.append({'path': study_path, 'subject_id': subject_id, 'nodule' : nodule_nr, 'scan_session' : subjects_session_numb_dict[subject_id], 'study_date' : subject_metadata[2], 'modality' : subject_metadata[4],
                            'study_description' : subject_metadata[3],'scan_description' : subject_metadata[5], 'manufacturer' : subject_metadata[6],'manufacturer_model' : subject_metadata[7], 'sofware_version' : subject_metadata[8],
                            'radiologists_lobular_score': radiologists_lobular_score,'radiologists_spiculation_score' : radiologists_spiculation_score, 'radiologists_subtlety_score' : radiologists_subtlety_score,
                            'radiologists_internal_structure_def' : radiologists_internal_structure_def,'radiologists_calcification_def' : radiologists_calcification_def , 'radiologists_sphericity_score' : radiologists_sphericity_score,
                            'radiologists_margin_score':radiologists_margin_score, 'radiologists_texture_def' : radiologists_texture_def, 'radiologists_malignancy_score' : radiologists_malignancy_score,
                            'median_malignancy_score' : median(radiologists_malignancy_score)} , ignore_index=True)

    df = df.loc[df.median_malignancy_score != 3].reset_index(drop=True)
    df['malignancy'] = 'benign'
    df.loc[df.median_malignancy_score > 3, 'malignancy'] = 'malignant'

    path_list = df.path.values
    for fn in path_list: 
        if not (DATA_DIR/'interim'/fn).exists(): 
            df = df.loc[df.path != fn].reset_index(drop=True)
    
    return df 
    
def _lidc_idri_split_train_test(df):
    """Split the dataframe into training and test based on patient-level diagnosis column. 
    
    Parameters
    ----------
    df : DataFrame
         Information about the nodules and patient-level diagnosis 
    
    Returns
    -------
    DataFrame
        Training data
    DataFrame 
        Test data
    """
    
    train_df = df[df.patient_level_diagnosis.isnull()].copy()
    train_df['usage'] = 'train'
    
    test_df = df[~df.patient_level_diagnosis.isnull()].copy()
    test_df['usage'] = 'test'
    
    return train_df, test_df
    
    
#TODO change name 
def download_and_preprocess_lidc_idri_tabular_data(): 
    """Download and preprocess the metadata and diagnosis data from LIDC-IDRI."""
    
    data_dest = Path(os.getcwd())/'..'/'local_data'/'interim'
    
    metadata_fn = download_data(URLs.LIDC_IDRI_METADATA, dest=data_dest, return_fn=True)
    diagnosis_fn = download_data(URLs.LIDC_IDRI_PATIENT_DATA, dest=data_dest, return_fn=True)
    
    metadata_df = _create_lidc_idri_dataframe(metadata_path=data_dest/metadata_fn)
    df = _add_patient_level_diagnosis(metadata_df,diagnois_data_path=data_dest/diagnosis_fn)
    
    train_df, test_df = _lidc_idri_split_train_test(df)
    
    processed_data_dir  = data_dest/'..'/'processed'
    processed_data_dir.mkdir(exist_ok=True)
    
    train_df.to_csv(processed_data_dir/'train_data_patient_level_diagnosis.csv', index=False)
    test_df.to_csv(processed_data_dir/'test_data_patient_level_diagnosis.csv', index=False)
    
    
    
    
    
    

