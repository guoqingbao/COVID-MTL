## Segmented CT Lung Volumes and Corresponding Clinical and Radio Features
Extracted radio features and corresponding clinical features were provided.

Compressed CT lung volumes will upload soon...

## Usage
The original CT scan with over 300 MB, the segmented CT lung volume with over 100 MB, the compressed CT lung volume with 5 MB (lossless).

## Compression and Decompression Algorithms
```python
# input: uncompressed numpy array; output: compressed DCM or DTM file
# note: DCM file is different from DICOM DCM file.
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

# input: compressed DCM or DTM file; output: decompressed numpy array
def GetData(srcFile):    
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
```
