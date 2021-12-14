import os
import numpy as np 
import pandas as pd
import numpy as np
import gc
import glob
from multiprocessing.dummy import Pool
from utils import GetCropped
from skimage import feature as ft
from skimage.measure import label,regionprops
gc.enable()
from utils import GetCropped
from sklearn.decomposition import PCA

class HOGFeatureExtractor:
    def __init__(self, tasks, dstCSV, isFeatureReduction = False):
        self.isFeatureReduction = isFeatureReduction
        self.dstCSV = dstCSV
        self.tasks = tasks
        self.allfeatureValues = {}

    def IdxOfLargestLungAreas(self, frames):
        lungAreas = 0
        largestIdx = 0
        for i in range(len(frames)):
            framelabel = label(frames[i] > 0)
            areas = [r.area for r in regionprops(framelabel)]
            if lungAreas < np.sum(areas):
                lungAreas = np.sum(areas)
                largestIdx = i
        return largestIdx

    def ExtractFeature(self, task):
        srcFile, pid = task
        image_s = GetCropped(srcFile)
        idx = self.IdxOfLargestLungAreas(image_s)
        frame = image_s[idx]
        frame = frame - np.min(frame)
        frame = frame / np.max(frame)
        features = ft.hog(frame,  # input image
                        orientations=32,  # number of bins
                        pixels_per_cell=(16, 16), # pixel per cell
                        cells_per_block=(1,1), # cells per blcok
                        block_norm = 'L1', #  block norm : str {‘L1’, ‘L1-sqrt’, ‘L2’, ‘L2-Hys’}, optional
                        transform_sqrt = True, # power law compression (also known as gamma correction)
                        feature_vector=True, # flatten the final vectors
                        visualize=False) # return HOG map
        self.allfeatureValues.update({pid:features.astype(np.float16)})

    def ParallelProcessing(self, num_of_parallel_tasks = 2):
        tasks = self.tasks.copy()
        num_processed = 0
        while len(tasks) > 0:
            if len(tasks) < num_of_parallel_tasks:
                subtasks = tasks[0:]
                tasks = []
            else:
                subtasks = tasks[:num_of_parallel_tasks]
                tasks = tasks[num_of_parallel_tasks:]

            # print("Processing Patients {0}".format(subtasks))
            with Pool(num_of_parallel_tasks) as p:
                p.map(self.ExtractFeature, subtasks)
                p.close()
                p.join()
                num_processed += len(subtasks)
                print("Processed NO. of Patients ", num_processed, "/", (len(self.tasks)))

        matframe = np.vstack(self.allfeatureValues.values())

        if self.isFeatureReduction:
            matframe = self.FeatureReduction(matframe)

        frame = pd.DataFrame(matframe, index=list(self.allfeatureValues.keys()))
        frame.to_csv(self.dstCSV)
        print('Write file ', self.dstCSV)

    def FeatureReduction(self, frame):
        pca = PCA(n_components=100)
        components = pca.fit_transform(frame.astype(np.float16))
        return components
          



if __name__ == "__main__":

    path = '/guoqing/project/covid/'
    dcmpath = path + "DCM/"
    outpath = path + "data/"
    files = [os.path.basename(file) for file in glob.glob(dcmpath +"*.dcm")]
    
    fileframe = pd.DataFrame({"File":files})
    pids = [int(f[f.rfind('-')+1:-4]) for f in fileframe.File]
    fileframe.loc[:, 'PID'] = pids
    fileframe.sort_index(inplace=True)

    dstCSV = outpath + 'hogfeatures.csv' # compressed as hogfeatures.zip

    tasks = []
    for idx, row in fileframe.iterrows():
        caseid = row['PID']
        file = row['File']
        tasks.append((dcmpath + file, caseid))

    extractor = HOGFeatureExtractor(tasks, dstCSV=dstCSV)
    extractor.ParallelProcessing(num_of_parallel_tasks = 5)
    





