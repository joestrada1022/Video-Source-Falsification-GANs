
import os


def get_prediction_files(input_dir):
    # Get all files (i.e. models) from input directory
    files_list = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and not f.startswith('.')]
    files_list = sorted(files_list)
    # print(f"Found {len(files_list)} files in {input_dir}: {files_list}")

    return files_list

def find_max_mode(list1):
    numeral=[[list1.count(nb), nb] for nb in list1]
    numeral.sort(key=lambda x:x[0], reverse=True)
    return(numeral[0][1])
    