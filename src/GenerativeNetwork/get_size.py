# import os, cv2

# def get_video_resolution(file_path):
#     try:
#         cap = cv2.VideoCapture(file_path)
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         cap.release()
#         return (width, height)
#     except Exception as e:
#         print("Error:", e)
#         return None

# def main():
#     downloads_dir = os.path.expanduser("/home/cslfiu/dev/cnn_vscf/dataset/Xiaomi/Training")
#     mp4_files = [f for f in os.listdir(downloads_dir) if f.endswith(".mp4") or f.endswith(".MOV")]
    
#     for file_name in mp4_files:
#         file_path = os.path.join(downloads_dir, file_name)
#         resolution = get_video_resolution(file_path)
#         if resolution:
#             print(f"File: {file_name}, Resolution: {resolution}")

# if __name__ == "__main__":
#     main()


# make a new script that gets the resolution of all the images in a folder
# and prints the resolution of each image with name

import os, cv2

def get_image_resolution(file_path):
    try:
        img = cv2.imread(file_path)
        return img.shape[:2]
    except Exception as e:
        print("Error:", e)
        return None
    
def main():
    downloads_dir = os.path.expanduser("/home/cslfiu/dev/cnn_vscf/frames/Galaxi-A50/Training/Samsung-A50-2(65)/")
    jpg_files = [f for f in os.listdir(downloads_dir) if f.endswith(".jpg")]
    
    for file_name in jpg_files:
        file_path = os.path.join(downloads_dir, file_name)
        resolution = get_image_resolution(file_path)
        if resolution:
            print(f"File: {file_name}, Resolution: {resolution}")

if __name__ == "__main__":
    main()