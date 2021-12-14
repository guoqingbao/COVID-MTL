import io
import cv2
import pydicom
import datetime
import numpy as np 
import os, glob
from pydicom.dataset import FileDataset, FileMetaDataset
from skimage import io
from pydicom.encaps import decode_data_sequence 
from pydicom.encaps import encapsulate
from PIL import Image
from multiprocessing import Pool

path = '/guoqing/project/covid/'

def SaveCompressedResults(frames, dstFile, PatientID):
    frame_data = []
    for frame in frames:
        image = Image.fromarray(frame.astype(np.ushort))
        with io.BytesIO() as output:
            image.save(output, format="PNG")
            frame_data.append(output.getvalue())

    encapsulated_data = encapsulate(frame_data)

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.1.2.4.51'
    file_meta.MediaStorageSOPInstanceUID = "1.2.3"
    file_meta.ImplementationClassUID = "1.2.3.4"

    ds = FileDataset(dstFile, {},
                     file_meta=file_meta, preamble=b"\0" * 128)

    ds.PatientName = "Patient " + str(PatientID)
    ds.PatientID = str(PatientID)

    # Set creation date/time
    dt = datetime.datetime.now()
    ds.ContentDate = dt.strftime('%Y%m%d')
    timeStr = dt.strftime('%H%M%S.%f') 
    ds.ContentTime = timeStr

    ds.PixelData = encapsulated_data
    from pydicom.uid import RLELossless
    ds.file_meta.TransferSyntaxUID = RLELossless

    print("Writing: " + dstFile)
    ds.save_as(dstFile)
    
def GetData(pid, dtm=True):    
    if dtm:
        srcFile = path + 'DTM/Patient-'+str(pid)+'.dtm'
    else:
        srcFile = path + 'DMM/Patient-'+str(pid)+'.dmm'

    dataset = pydicom.dcmread(srcFile)
    dataset.file_meta.TransferSyntaxUID
    dataset.BitsAllocated = 16
    frames = decode_data_sequence(dataset.PixelData)
    restored = []
    for f in frames:
        buf = io.BytesIO(f)
        img = cv2.imdecode(np.frombuffer(buf.getbuffer(), np.ushort), -1)
        restored.append(img)    
    restored = np.array(restored)
    return restored

def CropSave(pid):
    try:
        to=(340, 390, 390)
        restored = GetData(pid)
        start0 = int(abs(restored.shape[0]-to[0])/2)
        start1 = int(abs(restored.shape[1]-to[1])/2)
        start2 = int(abs(restored.shape[2]-to[2])/2)
        if restored.shape[0] >= to[0] and restored.shape[1]>=to[1] and restored.shape[2]>=to[2]:
            cropped = restored[start0:start0+to[0], start1:start1+to[1], start2:start2+to[2]]
        else:
            cropped = np.empty(to)
            if restored.shape[0] >= to[0]:
                cropped[:, start1:start1+restored.shape[1], start2:start2+restored.shape[2]] = restored[start0:start0+to[0], :, :]
            elif restored.shape[1]>=to[1]:
                cropped[start0:start0 + restored.shape[0], :, :] = restored[:, start1:start1+to[1], start2:start2+to[2]]
            else:
                cropped[start0:start0 + restored.shape[0], start1:start1+restored.shape[1], start2:start2 + restored.shape[2]] = restored
        dstFile = path + 'DCM/Patient-'+str(pid)+'.dcm'
        SaveCompressedResults(cropped, dstFile, pid)
    except:
        print("Bad input ", pid)

def CropVolumesParallel():
    files = [os.path.basename(file) for file in glob.glob(path +"DTM/*.dtm")]
    files.sort(key=lambda x: int(x[8:-4]))
    pids = [int(f[8:-4]) for f in files]

    files = [os.path.basename(file) for file in glob.glob(path +"DCM/*.dcm")]
    files.sort(key=lambda x: int(x[8:-4]))
    pids_processed = [int(f[8:-4]) for f in files]
    
    pidds = []
    for pid in pids:
        if not pid in pids_processed:
            pidds.append(pid)

    while len(pidds) > 0:
        if len(pidds) < 20:
            tasks = pidds[0:]
            pidds = []
        else:
            tasks = pidds[:20]
            pidds = pidds[20:]
        print("Processing Patients {0}".format(tasks))
        with Pool(15) as p:
            p.map(CropSave, tasks)
            p.close()
            p.join()

# %%
# Convert segmented lung volumes to same size (with cropped)
# CropVolumesParallel()

