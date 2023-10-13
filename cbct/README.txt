CBCT Face Anonymization tool-- Face de_identification.py
The main function in this file is 'predictions'.
Arguments: 
1) input_folder (folder that contains the .dcm files to be anonymized)
2) output_folder (folder where the resultant anonymized .dcm files will be saved)
Returns:
1) output_list (list of output paths of all anonymized dcm files)
2) classUID (list of classUIDs of all anonymized dcm files)
3) mimeType (dicom tag)
4)recommendation_string (the finding, conclusion, recommendation as dictionary)

It will check if the incoming scan is: CBCT, CT or otherwise and give results accordingly.