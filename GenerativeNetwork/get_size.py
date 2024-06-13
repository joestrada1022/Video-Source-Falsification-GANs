import os, cv2

def get_video_resolution(file_path):
    try:
        cap = cv2.VideoCapture(file_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return (width, height)
    except Exception as e:
        print("Error:", e)
        return None

def main():
    downloads_dir = os.path.expanduser("~/Downloads")
    mp4_files = [f for f in os.listdir(downloads_dir) if f.endswith(".mp4") or f.endswith(".MOV")]
    
    for file_name in mp4_files:
        file_path = os.path.join(downloads_dir, file_name)
        resolution = get_video_resolution(file_path)
        if resolution:
            print(f"File: {file_name}, Resolution: {resolution}")

if __name__ == "__main__":
    main()
