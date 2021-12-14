import glob
import numpy as np 
import pydicom
import os
import datetime
import io
import scipy.ndimage
import zipfile
from shutil import rmtree 
from multiprocessing import Pool
import scipy.ndimage as ndimage
from PIL import Image
from pydicom.encaps import decode_data_sequence 
from pydicom.encaps import encapsulate
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import RLELossless
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.morphology import disk, binary_erosion, binary_closing
from skimage.measure import label,regionprops
from skimage import measure
from skimage.segmentation import clear_border, active_contour
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class LungSegmentation:
    def __init__(self, tasks, outputPath, isDicom = True, nii_depthfactor = 2.0) -> None:
        self.tasks = tasks
        self.outputPath = outputPath
        self.isDicom = isDicom
        self.nii_depthfactor = nii_depthfactor

    def ReadDICOM(self, path):
            # Sort the dicom slices in their respective order
            files = [os.path.basename(file) for file in glob.glob(path +"*.dcm")]
            
            slices = [pydicom.read_file(path + filename) for filename in files]
            slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
            try:
                slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
            except:
                slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

            for s in slices:
                s.SliceThickness = slice_thickness.astype(np.float64)
            
            spacing = np.array([float(slices[0].SliceThickness), float(slices[0].PixelSpacing[0]), float(slices[0].PixelSpacing[1])] , dtype=np.float32)

            slices = np.stack([s.pixel_array for s in slices])
            slices[slices == -2000] = 0
            
            if np.max(slices) > 4095:
                print('range: ', np.min(slices), np.max(slices))
                if np.min(slices) < 0:
                    slices = slices + 1024
                slices[slices > 4095] = 4095
                slices[slices < 0] = 0
            elif np.min(slices) < -1024:
                print('range: ', np.min(slices), np.max(slices))
                slices = slices + 1024
                slices[slices > 4095] = 4095
                slices[slices < 0] = 0
            print('range: ', np.min(slices), np.max(slices))

            return slices, spacing
                
    def ReadNII(self, path):
            
            img = nib.load(path)
            spacing = img.header['pixdim'][:3]
            spacing[0] = abs(spacing[0]) * self.nii_depthfactor
    #         spacing = np.array([space[1], space[2], space[0]])
    #         img = nib.funcs.as_closest_canonical(img)
            slices = np.array(img.dataobj)    
        
            slices = np.moveaxis(slices, -1, 0)
            slices = np.rot90(slices, k = -1, axes = (1,2))

    #         slices = np.array(img.get_fdata())
            
            slices[slices == -2000] = 0
            
            if np.max(slices) > 4095:
                print('range: ', np.min(slices), np.max(slices))
                if np.min(slices) < 0:
                    slices = slices + 1024
                slices[slices > 4095] = 4095
                slices[slices < 0] = 0
            elif np.min(slices) < -1024:
                print('range: ', np.min(slices), np.max(slices))
                slices = slices + 1024
                slices[slices > 4095] = 4095
                slices[slices < 0] = 0
            print('range: ', np.min(slices), np.max(slices))

            return slices, spacing

    def resample(self, image, spacing, new_spacing=[1,1,1]):
        # Determine current pixel spacing

        resize_factor = spacing / new_spacing
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape
        new_spacing = spacing / real_resize_factor
        
        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
        
        return image, new_spacing

    #write compressed dicom
    def SaveCompressedResults(self, frames, dstFile, PatientID):
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

        ds.file_meta.TransferSyntaxUID = RLELossless

        print("Writing: " + dstFile)
        ds.save_as(dstFile)

    def plot_3d(self, image, threshold=-300):
        
        # Position the scan upright, 
        # so the head of the patient would be at the top facing the camera
        p = image.transpose(2,1,0)
        
        verts, faces = measure.marching_cubes_classic(p, threshold)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Fancy indexing: `verts[faces]` to generate a collection of triangles
        mesh = Poly3DCollection(verts[faces], alpha=0.70)
        face_color = [0.45, 0.45, 0.75]
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)

        ax.set_xlim(0, p.shape[0])
        ax.set_ylim(0, p.shape[1])
        ax.set_zlim(0, p.shape[2])

        plt.show()


    def largest_label_volume(self, im, bg=-1):
        vals, counts = np.unique(im, return_counts=True)

        counts = counts[vals != bg]
        vals = vals[vals != bg]

        if len(counts) > 0:
            return vals[np.argmax(counts)]
        else:
            return None

    #classic high-performance lung segmentation (from Kaggle)
    def segment_lung_mask(self, image, fill_lung_structures=True):
        
        # not actually binary, but 1 and 2. 
        # 0 is treated as background, which we do not want
        # binary_image = np.array(image > 604, dtype=np.int8)+1
        binary_image = (image < 604).astype('int8')
        
        # Remove the blobs connected to the border of the image
        for i in range(len(binary_image)):
            binary_image[i] = clear_border(binary_image[i])
        binary_image = np.invert(binary_image).astype(np.int8)

    #     plt.imshow(binary_image[150], cmap='gray')
    #     plt.show()
        labels = measure.label(binary_image)
        
        # Pick the pixel in the very corner to determine which label is air.
        #   Improvement: Pick multiple background labels from around the patient
        #   More resistant to "trays" on which the patient lays cutting the air 
        #   around the person in half
        background_label = labels[0,0,0]
        
        #Fill the air around the person
        binary_image[background_label == labels] = 2
        
        background_label = labels[image.shape[0]-1,image.shape[1]-1,image.shape[2]-1]
        
        #Fill the air around the person
        binary_image[background_label == labels] = 2
        
        # Method of filling the lung structures (that is superior to something like 
        # morphological closing)
        if fill_lung_structures:
            # For every slice we determine the largest solid structure
            for i, axial_slice in enumerate(binary_image):
                axial_slice = axial_slice - 1
                labeling = measure.label(axial_slice)
                l_max = self.largest_label_volume(labeling, bg=0)
                
                if l_max is not None: #This slice contains some lung
                    binary_image[i][labeling != l_max] = 1

        
        binary_image -= 1 #Make the image actual binary
        
    #     plt.imshow(binary_image[150], cmap='gray')
    #     plt.show()
        
        binary_image = 1-binary_image # Invert it, lungs are now 1
    #     plt.imshow(binary_image[150], cmap='gray')
    #     plt.show()
        # Remove other air pockets insided body
        labels = measure.label(binary_image, background=0)
        l_max = self.largest_label_volume(labels, bg=0)
        if l_max is not None: # There are air pockets
            binary_image[labels != l_max] = 0
    
        return binary_image

    def SaveLungAreas(self, frames, dstFile):
        lungAreas = []
        for i in range(len(frames)):
            framelabel = label(frames[i] > 0)
            areas = [r.area for r in regionprops(framelabel)]
            lungAreas.append(np.sum(areas))
        fig = plt.figure()
        fig.set_size_inches(10, 3)

        plt.plot(lungAreas)
        fig.tight_layout()
        fig.savefig(dstFile)

    def TransBackground(self, img):
        imgg = img.copy()
        pixels = imgg.load() 
        for i in range(imgg.size[0]): 
            for j in range(imgg.size[1]):
                if pixels[i,j] == (0, 0, 0, 255):
                    pixels[i,j] = (0, 0 ,0, 0)
                else:
                    pixels[i,j] = (int(pixels[i,j][0]), int(pixels[i,j][1]),pixels[i,j][2],int(0.3*255))
        return imgg

    #perform segmentation refinement
    def SegRefine(self, seg, ct):
        refined = []
        pg = 0
        for sliceid in range(len(seg)):
            binary = seg[sliceid].copy()
            framelabel = label(binary)
            rprobas = regionprops(framelabel)
            areas = [r.area for r in rprobas]
            areas.sort()
            originImg = ct[sliceid].copy()

            if len(areas) < 2 or np.sum(areas) < 15000:
                binary = binary_closing(binary, disk(5))
                refined.append(binary.astype('int8'))
                pgress = int(sliceid/len(seg)*100)
                if  pgress != pg and pgress % 5 ==0:
                    pg = pgress
                    print("Progress {0}%".format(pgress))
                continue

            lunglst = [np.zeros_like(binary, dtype='bool'), np.zeros_like(binary, dtype='bool')]

            for region in rprobas:
                if region.area == areas[-1]:
                    lunglst[0][region.coords[:, 0], region.coords[:, 1]] = 1
                else:
                    lunglst[1][region.coords[:, 0], region.coords[:, 1]] = 1

            snakes = []
            mask = np.zeros_like(binary, dtype='bool')
            for lung in lunglst:
                bins = binary_closing(lung, disk(10))
                contours = measure.find_contours(bins, 0.8)
                lungimg = originImg.copy()
                pixelremoved = lung == 0
                lungimg[pixelremoved] = 0
                for contour in contours:
                    snake = active_contour(lungimg, contour,alpha=0.02,beta=20, max_iterations=20, coordinates='rc')
                    r_mask = np.zeros_like(binary, dtype='bool')
                    r_mask[snake[:, 0].astype('int'), snake[:, 1].astype('int')] = 1
                    r_mask = ndimage.binary_fill_holes(r_mask)
                    mask = mask | r_mask

            masksub = mask.astype(np.uint8) - binary.astype(np.uint8)   
            binarywithsub = (binary.astype('bool') | masksub.astype('bool')).astype(np.uint8) 
            
            for lung in lunglst:   
                lung[lung>0] = 0

            rprobas = regionprops(label(binarywithsub))
            areas = [r.area for r in rprobas]
            areas.sort()

            for region in rprobas:
                if region.area == areas[-1]:
                    lunglst[0][region.coords[:, 0], region.coords[:, 1]] = 1
                elif region.area == areas[-2]:
                    lunglst[1][region.coords[:, 0], region.coords[:, 1]] = 1

            binary[binary>0] = 0
            for lung in lunglst:        
                lung = binary_erosion(lung, disk(2))
                lung = binary_closing(lung, disk(10))
                contours = measure.find_contours(lung, 0.8)
                binary = binary.astype('bool') | lung.astype('bool')

            refined.append(binary.astype('int8'))

            pgress = int(sliceid/len(seg)*100)
            if  pgress != pg and pgress % 5 ==0:
                pg = pgress
                print("Progress {0}%".format(pgress))
        return np.array(refined)

    def unzip(self, filename, to_path):
        zipFile = zipfile.ZipFile(filename)
        for file in zipFile.namelist():
            zipFile.extract(file, to_path)
        zipFile.close()

    def ProcessDICOM(self, task):
        # inputPath = PATH + 'Patient '+str(CaseID)+'/CT/'
     
        # print("Read Patient CT ", CaseID)
        file, caseid = task
        try:
            ctframes, spacing = self.ReadDICOM(file)
        except Exception as e:
            print("Error when reading patient ", caseid)
            print(repr(e))
            return "Case " + str(caseid) + " Exception!"
        return self.ProcessPatient(ctframes, spacing, caseid)

    def ProcessNII(self, task):
        # inputPath = PATH + 'Patient '+str(CaseID)+'/CT/'
     
        # print("Read Patient CT ", caseid)
        file, caseid = task
        try:
            ctframes, spacing = self.ReadNII(file)
        except Exception as e:
            print("Error when reading patient ", caseid)
            print(repr(e))
            return "Case " + str(caseid) + " Exception!"
        return self.ProcessPatient(ctframes, spacing, caseid)

    def ProcessPatient(self, ctframes, spacing, CaseID):

        fig = plt.figure()
        fig.set_size_inches(7, 4)
        plt.hist(ctframes.flatten(), bins=80, color='c')
        plt.xlabel("Hounsfield Units (HU)")
        plt.ylabel("Frequency")
        fig.tight_layout()
        dstFile = self.outputPath + 'HU/Patient-' + str(CaseID) + '.png'
        fig.savefig(dstFile)


        ctresampled, newspacing = self.resample(ctframes, spacing, [1,1,1])
        ctresampled[ctresampled<0] = 0
        print("Shape before resampling\t", ctframes.shape)
        print("Shape after resampling\t", ctresampled.shape)
        # segmented_lungs = segment_lung_mask(ctresampled, False)
        segmented_lungs = self.segment_lung_mask(ctresampled, True)

        dstFile = self.outputPath + 'lungAreasInit/Patient-' + str(CaseID) + '.png'

        self.SaveLungAreas(segmented_lungs, dstFile)

        lungFrames = self.SegRefine(segmented_lungs, ctresampled)

        # save mask
        dstFile = self.outputPath + 'DMM/Patient-'+str(CaseID)+'.dmm'
        self.SaveCompressedResults(lungFrames, dstFile, CaseID)

        # save ct lung
        pixelremoved = lungFrames == 0
        ctresampled[pixelremoved] = 0
        dstFile = self.outputPath + 'DTM/Patient-'+str(CaseID)+'.dtm'
        self.SaveCompressedResults(ctresampled, dstFile, CaseID)

        dstFile = self.outputPath + 'lungAreasFinal/Patient-' + str(CaseID) + '.png'
        self.SaveLungAreas(lungFrames, dstFile)
        return "Case " + str(CaseID) + " Processed."

    def ParallelProcessing(self, needUnzip = True):
        tasks = self.tasks
        while len(tasks) > 0:
            if len(tasks) < 10:
                subtasks = tasks[0:]
                tasks = []
            else:
                subtasks = tasks[:10]
                tasks = tasks[10:]

            # in case the file is zipped, unzip it first    
            for i in range(len(subtasks)):
                file, caseid = subtasks[i]
                if needUnzip and not os.path.exists(self.outputPath + 'Patient '+str(caseid)+'/CT/'):
                    if os.path.exists(file):
                        self.unzip(file, self.outputPath)
                        print('unzipped '+ file)
                        subtasks[i] = self.outputPath + 'Patient '+str(caseid)+'/CT/'

            print("Processing Case {0}".format(subtasks))
            with Pool(10) as p:
                if self.isDicom:
                    print(p.map(self.ProcessDICOM, subtasks))
                else:
                    print(p.map(self.ProcessNII, subtasks))
                p.close()
                p.join()
            #in case you want to remove unzipped temp files
            # for task in tasks:
            #     dstFile = self.rootPath + 'DTM/Patient-'+str(task)+'.dtm'
            #     if os.path.exists(dstFile):
            #         rmtree(self.rootPath + 'Patient '+str(task)+'/')        


if __name__ == "__main__":
    #####################################Zipped DICOM files ########################
    PATH = '/guoqing/project/covid/'
    files = [os.path.basename(file) for file in glob.glob(PATH +"*.zip")]
    files.sort(key=lambda x: int(x[8:-4]))
    tasks = []
    for f in files:
        CaseID =int(f[8:-4])
        dstFile = PATH + 'DTM/Patient-'+str(CaseID)+'.dtm'
        if os.path.exists(dstFile):
            if os.path.exists(PATH + 'Patient '+str(CaseID)+'/CT/'):
                rmtree(PATH + 'Patient '+str(CaseID)+'/')
                print("Remove Processed Folder ", PATH + 'Patient '+str(CaseID)+'/')
            continue

        dst = '/home/gbao5100/project/covid/Patient-{0}.zip'
        tasks.append((dst, CaseID))

    ################################ NII Files without CaseID ###########################
    if False:
        PATH = '/guoqing/project/covid/NII-OUT/'
        files = [os.path.basename(file) for file in glob.glob("/guoqing/project/covid/NII/*.gz")]
        tasks = []
        for i in range(len(files)):
            tasks.append((f, 2000 + i)) # file, and caseid

    segmentor = LungSegmentation(tasks=tasks, outputPath = PATH, isDicom=True)
    segmentor.ParallelProcessing(needUnzip=True)
