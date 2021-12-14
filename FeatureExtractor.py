import os
import csv
import SimpleITK as sitk
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import gc
import glob
from PIL import Image, ImageDraw
from radiomics import firstorder, glcm, shape, glrlm, glszm, featureextractor
from multiprocessing.dummy import Pool
from utils import GetCropped

gc.enable()

FEATURE_TYPES = {1:"shape", 2:"firstorder", 3:"glcm", 4:"glrlm", 5:"glszm", 6:"wavelet-glcm", 7:"wavelet-glrlm"}
TASK_LIST = [2,3,4,5,6,7]

class FeatureExtractor:
    def __init__(self, tasks, yamlpath, dstCSV) -> None:
        self.yamlpath = yamlpath
        self.dstCSV = dstCSV
        self.tasks = tasks
        self.settings = {}
        self.spacing = ['1.0','1.0','1.0']  
        self.settings['binWidth'] = 25
        self.settings['resampledPixelSpacing'] = None  
        self.settings['interpolator'] = 'sitkBSpline'
        self.settings['verbose'] = True

    def ExtractFeature(self, task):
        srcFile, pid = task
    #     try:
        image_s = GetCropped(srcFile)
    #     print(pid, ": Readed Patient")
        mask_s = image_s.copy()
        mask_s[mask_s > 0] = 1
        image = sitk.GetImageFromArray(image_s)
        mask = sitk.GetImageFromArray(mask_s)
        del image_s, mask_s

        allfeatureValues = []
        for FEATURE_ID in TASK_LIST:
    #         print(pid, ": Extract feature ", FEATURE_TYPES[FEATURE_ID])

            if FEATURE_ID == 6:
                paramPath = self.yamlpath + 'Params-glcm.yaml'
            else:
                paramPath = self.yamlpath + 'Params-glrlm.yaml'

            selected_feature = None
            if FEATURE_TYPES[FEATURE_ID] == "firstorder":
                extractor = firstorder.RadiomicsFirstOrder(image, mask, **self.settings)
            elif FEATURE_TYPES[FEATURE_ID] == 'shape':
                extractor = shape.RadiomicsShape(image, mask, **self.settings)
            elif FEATURE_TYPES[FEATURE_ID] == "glcm":
                extractor = glcm.RadiomicsGLCM(image, mask, **self.settings)
            elif FEATURE_TYPES[FEATURE_ID] == "glrlm":
                extractor = glrlm.RadiomicsGLRLM(image, mask, **self.settings)
            elif FEATURE_TYPES[FEATURE_ID] == "glszm":
                extractor = glszm.RadiomicsGLSZM(image, mask, **self.settings)
            elif FEATURE_TYPES[FEATURE_ID] == "wavelet-glcm" or FEATURE_TYPES[FEATURE_ID] == "wavelet-glrlm":
                extractor = featureextractor.RadiomicsFeatureExtractor(paramPath)
            else:
                raise Exception("Invalid feature selected!")

            featureValues= {}
            if FEATURE_ID == 6 or FEATURE_ID == 7:
                featureValues = extractor.execute(image,mask)   
            elif extractor != None:
                extractor.enableAllFeatures()
                extractor.execute()
                featureValues = extractor.featureValues
            else:
                raise Exception("Invalid feature configuration!")
    #         print(pid, ": Extracted feature ", FEATURE_TYPES[FEATURE_ID])
            print(FEATURE_TYPES[FEATURE_ID], " number of features: ", len(featureValues))
            allfeatureValues.append(featureValues)
        del image, mask
        gc.collect()
        values = {'PID':pid}
        for featureValues in allfeatureValues:
            values.update(featureValues)
        print(pid, ": Processed")
    #     except Exception as e:
    #         print("Exception ", e)
        retvalues = {k.replace('wavelet-',''):v for k, v in values.items() if not k.startswith('general_')}

        return {k:v for k, v in retvalues.items() if not k.startswith('diagnostics_')}

    def ParallelProcessing(self, num_of_parallel_tasks = 2):
        tasks = self.tasks
        while len(tasks) > 0:
            if len(tasks) < num_of_parallel_tasks:
                subtasks = tasks[0:]
                tasks = []
            else:
                subtasks = tasks[:num_of_parallel_tasks]
                tasks = tasks[num_of_parallel_tasks:]

            print("Processing Patients {0}".format(subtasks))
            with Pool(num_of_parallel_tasks) as p:
                records = p.map(self.ExtractFeature, subtasks)
                p.close()
                p.join()
                if not os.path.exists(self.dstCSV):
                    with open(self.dstCSV, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=records[0].keys())
                        writer.writeheader()
                        f.flush()
                        f.close()
                processed = []
                with open(self.dstCSV, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=records[0].keys())
                    for record in records:
                        writer.writerow(record)
                        processed.append(record['PID'])
                    f.flush()
                    f.close()
                print("Processed Patients ", processed)



if __name__ == "__main__":

    path = '/guoqing/project/covid/'
    dcmpath = path + 'DCM/'
    outpath = path + 'data/'
    files = [os.path.basename(file) for file in glob.glob(dcmpath +"*.dcm")]
    
    fileframe = pd.DataFrame({"File":files})
    pids = [int(f[f.rfind('-')+1:-4]) for f in files]
    fileframe.loc[:, 'PID'] = pids
    fileframe = fileframe.sort_values(by='PID').reset_index(drop=True)

    dstCSV = outpath + 'radiofeatures.csv'
    finished = []
    if os.path.exists(dstCSV):
        frame = pd.read_csv(dstCSV)
        finished.extend(list(frame.PID))
        print("Finished ", list(frame.PID))
        
    tasks = []
    for idx, row in fileframe.iterrows():
        caseid = row['PID']
        file = row['File']
        if caseid not in finished: 
            tasks.append((dcmpath + file, caseid))


    extractor = FeatureExtractor(tasks, path, dstCSV=dstCSV)
    extractor.ParallelProcessing()





