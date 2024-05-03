import os
import shutil
from zipfile import ZipFile
from google.colab import drive
import os


#zip_path is the path to the CityScapes zip in google drive and destination_path is the path to put the download
def Download(zip_path, destination_path):

    #Connect to google drive
    drive.mount('/content/drive')

    #Extraction
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(destination_path)

    print("The zip file has been extracted correctly")

def Modified_CityScapes(start_path):
    # Extract images and copy
    end_path = ['/gtFine/train', '/gtFine/val', '/images/train', '/images/val']
    for str in end_path:
        origin = start_path + str
        for subdir in os.listdir(origin):
            path_subdir = os.path.join(origin, subdir)
            if os.path.isdir(path_subdir):
                for file in os.listdir(path_subdir):
                    path_file_origin = os.path.join(path_subdir, file)
                    shutil.copy(path_file_origin, origin)

        # Delete subdirectory
        for subdir in os.listdir(origin):
            path_subdir = os.path.join(origin, subdir)
            if os.path.isdir(path_subdir):
                shutil.rmtree(path_subdir)