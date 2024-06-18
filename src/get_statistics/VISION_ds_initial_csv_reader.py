import argparse
import csv, os

from utility import get_prediction_files

parser = argparse.ArgumentParser(
    description='Make predictions with signature network',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument('--main_path', type=str, required=True, help='Path to directory consisting of .h5-models (to use for predicting)')
parser.add_argument('--csvs_folder', type=str, required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    main_path = args.main_path
    csvs_folder = args.csvs_folder

    path_to_csvs = os.path.join(main_path, csvs_folder)

    predicted_files = get_prediction_files(path_to_csvs)
    print(predicted_files)

    output_folder = os.path.join(main_path,"statistics")

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    files_path = os.path.join(output_folder, "initial_stats")
    if not os.path.exists(files_path):
        os.mkdir(files_path)




    predicted_files = sorted(predicted_files)
    
    for i, file in enumerate(predicted_files):
        path = os.path.join(path_to_csvs, file)
        print(file)

        VIDEO_NAMES = []
        # all_file_names = []
        with open(path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            # for row in reader:
            #     file_path, file_name = os.path.split(row["File"])
            #     video_name, _ = file_name.split(")")
            #     VIDEO_NAMES.append(video_name)


            # 1. get videos names
            # VIDEO_DIR = "/Users/marynavek/Projects/files/Videos"
            # DEVICES = [item for item in os.listdir(VIDEO_DIR) if os.path.isdir(os.path.join(VIDEO_DIR, item))]

            FILTERED_RESULTS = []
            
            newreader = []
            for row in reader:
                file_name_row = row["File"]
                # file_name_row = file_name_row.replace("['", "")
                # file_name_row = file_name_row.replace("']", "")
                # print(file_name_row)
                # remove_part, video_name3 = file_name_row.split("_vid_name")
                # video_name1 = video_name3.split(".")[0]
                # # print(video_name1)
                # video_name1 = video_name1.split("D")[1]
                # # print(video_name1)
                # patches_vide_name_arr = video_name1.split("_")

                file_path, file_name = os.path.split(file_name_row)
                video_name, _ = file_name.split(")")

                # video_name = ""
                # print(patches_vide_name_arr)
                # for d in range(len(patches_vide_name_arr)):
                #     if d != len(patches_vide_name_arr)-1:
                #         video_name += patches_vide_name_arr[d]
                # print(video_name)
                row["Video Name"] = video_name + ")"
                newreader.append(row)
                

            # print(newreader)
        file_name = 'output_frames_stats_' + str(i) + '_.csv'
        csv_path = os.path.join(files_path, file_name)
        with open(csv_path, 'w') as csvfile:
            fieldnames = reader.fieldnames
            fieldnames.append("Video Name")
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for data in newreader:
                writer.writerow(data)



