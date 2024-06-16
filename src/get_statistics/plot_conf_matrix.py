import csv, os
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from utility import get_prediction_files

path_to_csv = '/Users/marynavek/Projects/cnn_ensemble_vsi/statistics/most_common_label_by_video'

previous_files = get_prediction_files(path_to_csv)

output_folder = "/Users/marynavek/Projects/cnn_ensemble_vsi/statistics/"

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

files_path = os.path.join(output_folder, "matrix_identified_videos_per_device_complete")
if not os.path.exists(files_path):
    os.mkdir(files_path)

DEVICE_TYPES = [
        "Huawei-Y7Prime2019",
        "Huawei-Y9",
        "Iphone-8Plus", 
        "Nokia-5dot4", 
        "Nokia-7dot1", 
        "Samsung-A50", 
        "Samsung-Note9", 
        "Xiaomi-redmiNote8", 
        "Xioami-RedmiNote9Pro", 
        "iPhone-xsMax"
    ]

# VIDEO_TYPES = [
#         "flat",
#         "indoor",
#         "outdoor"
#     ]

for i in range(len(previous_files)):
    for file in previous_files:
        create_string = "_" + str(i) + '_'
        if create_string in file:
            path = os.path.join(path_to_csv, file)
            with open(path, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                sorted_reader = sorted(reader, key = lambda item: item['Video Name']) 
                main_stats_array_total = []
                for device_type in DEVICE_TYPES:
                    print(device_type)
                    device1 = 0
                    device2 = 0
                    device3 = 0
                    device4 = 0
                    device5 = 0
                    device6 = 0
                    device7 = 0
                    device8 = 0
                    device9 = 0
                    device10 = 0

                    device_array_total = []

                    for row in sorted_reader:
                        if device_type in row["Video Name"]:
                            if row["Mostly predicted label"] == "0":
                                device1 += 1
                            elif row["Mostly predicted label"] == "1":
                                device2 += 1
                            elif row["Mostly predicted label"] == "8":
                                device3 += 1
                            if row["Mostly predicted label"] == "2":
                                device4 += 1
                            elif row["Mostly predicted label"] == "3":
                                device5 += 1
                            elif row["Mostly predicted label"] == "4":
                                device6 += 1
                            elif row["Mostly predicted label"] == "5":
                                device7 += 1
                            elif row["Mostly predicted label"] == "6":
                                device8 += 1
                            if row["Mostly predicted label"] == "7":
                                device9 += 1
                            elif row["Mostly predicted label"] == "9":
                                device10 += 1
                    device_array_total = [device1, device2, device3, device4, device5, device6, device7, device8, device9, device10]
                    print(device_array_total)
                    main_stats_array_total.append(device_array_total)


            df_cm_total = pd.DataFrame(main_stats_array_total, range(10), range(10))
            print(df_cm_total)
            # plt.tight_layout(pad=2.0)

            plt.title('Total Videos')
            sn.set(font_scale=1) # for label size
            sn.heatmap(df_cm_total, annot=True, annot_kws={"size": 10}, cmap="Greys")

            # plt.subplot(1,4,2)
            # plt.title('Flat Videos')
            # sn.set(font_scale=1) # for label size
            # sn.heatmap(df_cm_flat, annot=True, annot_kws={"size": 10}, cmap="Greys", cbar=False, yticklabels=False) # font size

            # plt.subplot(1,4,3)
            # plt.title('Indoor Videos')
            # sn.set(font_scale=1) # for label size
            # sn.heatmap(df_cm_indoor, annot=True, annot_kws={"size": 10}, cmap="Greys", cbar=False, yticklabels=False) # font size

            # plt.subplot(1,4,4)
            # plt.title('Outdoor Videos')
            # sn.set(font_scale=1) # for label size
            # sn.heatmap(df_cm_outdoor, annot=True, annot_kws={"size": 10}, cmap="Greys", cbar=False, yticklabels=False) # font size

            file_name = 'matrix_identified_videos_per_device_' + str(i) + '_.png'
            matrix_path = os.path.join(files_path, file_name)
            # plt.show()
            plt.savefig(matrix_path)
            plt.savefig(matrix_path, format='png')
