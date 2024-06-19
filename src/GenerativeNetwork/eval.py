from utils import display_samples, apply_cfa
import datetime, cv2

id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
save_path = f"generated/sample-{id}.jpg"
save_path = None

image_path = 'data/frames/Galaxi-A50/Validation/Samsung-A50-2(299)/Samsung-A50-2(299).mp4_011.jpg'
image_path = None

display_samples("generated/models/gan20240618-211659/gen-24.keras", image_path=image_path, save_path=save_path)

# img = cv2.imread(image_path)
# img = apply_cfa(img)
# cv2.imwrite(f"generated/sample-{id}-original.jpg", img)