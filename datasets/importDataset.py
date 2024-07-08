import os
import shutil

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
