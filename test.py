import os 
import shutil

os.makedirs("datasets/data2", exist_ok=True)

for x in range(1000):
    source = f"datasets/data/{x:03d}.jpg.json"
    dest = f"datasets/data2/{x:03d}.jpg.json"
    if os.path.exists(source):
        shutil.copy(source, dest)
        