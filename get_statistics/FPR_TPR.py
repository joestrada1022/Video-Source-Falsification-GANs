import argparse
import csv, os
import numpy as np
import matplotlib.pyplot as plt
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

    files_path = os.path.join(output_folder, "fpr_tpr_stats")
    if not os.path.exists(files_path):
        os.mkdir(files_path)

# True Positive (TP): True label = A and predicted label = A
# False Positive (FP): True label != A and predicted label = A
# False Negative (FN): True label = A and predicted label != A
# True Negative (TN): True label != A and predicted label != A
# accuracy (ACC) = (TP + TN)/(TP + FN +TN +FP)
# false positive rate (FPR) = FP/(FP +TN)
# true positive rate (TPR) = TP/(TP+FN)
    DEVICE_TYPES = [
        "01SamsungGalaxyS3",
        "02HuaweiP9",
        "03AppleiPhone5c",
        "04AppleiPhone6",
        "05HuaweiP9Lite", 
        "06AppleiPhone6Plus",
        "07SamsungGalaxyS5",
        "08AppleiPhone5",
        "09HuaweiP8",
        "10SamsungGalaxyS4Mini"
    ]
    # DEVICE_TYPES = [
    #     "01AppleiPhone5c",
    #     "02HuaweiP9Lite",
    #     "03AppleiPhone6Plus",
    #     "04SonyXperiaZ1Compact",
    #     "05XiaomiRedmiNote3", 
    #     "06OnePlusA3003"
    # ]

    for i in range(len(previous_files)):
        for file in previous_files:
            create_string = "_" + str(i) + '_'
            if "output_frames_stats_0_" in file:
                path = os.path.join(path_to_csv, file)
                with open(path, newline='') as csvfile:
                    out_put_dictionary = []
                    reader = csv.DictReader(csvfile)
                    sorted_reader = sorted(reader, key = lambda item: item['Video Name']) 

                    TP_total = 0
                    FP_total = 0
                    FN_total = 0
                    TN_total = 0
                    FPR_total = 0
                    TPR_total = 0 
                    ACC_total = 0
                    PPV_total = 0
                    recal_total = 0
                    F1_total = 0
                    
                    for i, device in enumerate(DEVICE_TYPES):
                        print(device)
                        TP = 0
                        FP = 0
                        FN = 0
                        TN = 0
                        FPR = 0
                        TPR = 0 
                        ACC = 0
                        PPV = 0
                        recall = 0
                        F1 = 0
                        ground_true_label = str(i)
                        print(device)
                        print(ground_true_label)
                        
                        for row in sorted_reader:
                            true_label = row["True Label"]
                            predicted_label = row["Predicted Label"]
                            
                            if true_label == ground_true_label and predicted_label == ground_true_label:
                                TP += 1
                            elif true_label != ground_true_label and predicted_label == ground_true_label:
                                FP += 1
                            elif true_label == ground_true_label and predicted_label != ground_true_label:
                                FN += 1
                            elif true_label != ground_true_label and predicted_label != ground_true_label:
                                TN += 1
                        FPR = FP/(FP + TN)
                        TPR = TP/(TP+FN) 
                        ACC = (TP + TN)/(TP + FN +TN +FP)
                        PPV = TP / (TP + FP)
                        recall = TP / (TP + FN)
                        F1 = 2* ((recall*PPV)/(recall+PPV))
                        TP_total += TP
                        TN_total += TN
                        FP_total += FP
                        FN_total += FN
                        addValue = {"Device": device, "TPR": TPR, "FPR": FPR, "ACC":ACC, "PPV": PPV, "recall": recall, "F1": F1}
                        out_put_dictionary.append(addValue) 
                    FPR_total = FP_total/(FP_total +TN_total)
                    TPR_total = TP_total/(TP_total+FN_total) 
                    ACC_total = (TP_total + TN_total)/(TP_total + FN_total + TN_total + FP_total)
                    PVV_total = TP_total/(TP_total+FP_total)
                    recal_total = TP_total/(TP_total+FN_total)
                    F1_total = 2* ((recal_total*PVV_total)/(recal_total+PVV_total))
                    addValue = {"Device": "Total", "TPR": TPR_total, "FPR": FPR_total, "ACC":ACC_total, "PPV": PVV_total, "recall": recal_total, "F1": F1_total}
                    out_put_dictionary.append(addValue)


                    print(addValue)
                file_name = 'FPR_TPR_stats_' + str(i) + '_.csv'
                csv_path = os.path.join(files_path, file_name)

                # auc = -1 * np.trapz(TPR, FPR)

                # plt.plot(FPR, TPR, linestyle='--', marker='o', color='darkorange', lw = 2, label='ROC curve', clip_on=False)
                # plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
                # plt.xlim([0.0, 1.0])
                # plt.ylim([0.0, 1.0])
                # plt.xlabel('False Positive Rate')
                # plt.ylabel('True Positive Rate')
                # plt.title('ROC curve, AUC = %.2f'%auc)
                # plt.legend(loc="lower right")
                # plt.savefig('AUC_example.png')
                # plt.show()

                with open(csv_path, 'w') as csvfile:
                        fieldnames = ["Device", "TPR", "FPR", "ACC", "PPV", "recall", "F1"]
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        for data in out_put_dictionary:
                            writer.writerow(data)

                break                    