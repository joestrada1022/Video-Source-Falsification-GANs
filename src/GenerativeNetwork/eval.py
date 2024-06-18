from utils import display_samples
import datetime

id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
save_path = f"generated/sample-{id}.jpg"
# save_path = None
display_samples("generated/models/gan20240618-201219/gen-1.keras", save_path=save_path)