import argparse
import csv, os

from utility import find_max_mode, get_prediction_files

parser = argparse.ArgumentParser(
    description='Make predictions with signature network',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument('--main_path', type=str, required=True, help='Path to directory consisting of .h5-models (to use for predicting)')
parser.add_argument('--csvs_folder', type=str, required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    main_path = args.main_path
    csvs_folder = args.csvs_folder

    path_to_csv = os.path.join(main_path, csvs_folder, "initial_stats")

    previous_files = get_prediction_files(path_to_csv)

    output_folder = os.path.join(main_path, csvs_folder)

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    files_path = os.path.join(output_folder, "most_common_label_by_video")
    if not os.path.exists(files_path):
        os.mkdir(files_path)


    for i in range(len(previous_files)):
        for file in previous_files:
            create_string = "_" + str(i) + '_'
            if create_string in file:

                path = os.path.join(path_to_csv, file)
                with open(path, newline='') as csvfile:
                    reader = csv.DictReader(csvfile)
                    video_name = ""
                    new_video_name = ""
                    out_put_dictionary = []
                    number_of_frames = 0
                    predicted_labels = []
                    true_label = 0
                    most_common_label = 0
                    sorted_reader = sorted(reader, key = lambda item: item['Video Name']) 
                    for row in sorted_reader:

                        if not video_name == row["Video Name"]:
                            if not number_of_frames == 0:
                                most_common_label = find_max_mode(predicted_labels)
                                addValue = {"Video Name": video_name, "Total Frames": number_of_frames, "True Label": true_label, "Mostly predicted label": most_common_label}
                                out_put_dictionary.append(addValue)
                            true_label = row["True Label"]
                            number_of_frames = 0
                            number_of_correct_frames = 0
                            predicted_labels = []
                            most_common_label = 0
                            video_name = row["Video Name"]

                        number_of_frames += 1

                        predicted_labels.append(row["Predicted Label"])


                file_name = 'output_most_common_by_videos_stats_' + str(i) + '_.csv'
                csv_path = os.path.join(files_path, file_name)

                with open(csv_path, 'w') as csvfile:
                        fieldnames = ["Video Name", "Total Frames", "True Label", "Mostly predicted label"]
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        for data in out_put_dictionary:
                            writer.writerow(data)
                break
                


                
