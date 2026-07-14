import os
import shutil
from pathlib import Path
import pandas as pd

def convert_visa_to_mvtec(visa_root, target_root):

    Path(target_root).mkdir(exist_ok=True)
    csv_data = pd.read_csv(f'{visa_root}/split_csv/1cls.csv', header=0)
    columns = csv_data.columns  # [object, split, label, image, mask]

    for obj_dir in Path(visa_root).iterdir():
        if not obj_dir.is_dir():
            continue
        object_name = obj_dir.name
        obj_data = csv_data[csv_data['object'] == object_name]
        print(f"Processing object: {object_name}")
        
        target_obj_dir = Path(target_root) / object_name
        target_obj_dir.mkdir(exist_ok=True)

        normal_train_path = (target_obj_dir / "train/good")
        normal_train_path.mkdir(parents=True, exist_ok=True)
        
        normal_test_path = (target_obj_dir / "test/good")
        normal_test_path.mkdir(parents=True, exist_ok=True)
        
        anomaly_test_path = (target_obj_dir / "test/bad")
        anomaly_test_path.mkdir(parents=True, exist_ok=True)
        
        anomaly_mask_path = (target_obj_dir/ "ground_truth/bad")
        anomaly_mask_path.mkdir(parents=True, exist_ok=True)

        for _, row in obj_data[obj_data['split'] == 'train'].iterrows():
            image_path = os.path.join(visa_root , row['image'])
            # mask_path = obj_dir / row['mask']
            shutil.copy(image_path, normal_train_path)
            # shutil.copy(mask_path, normal_train_path)

        for _, row in obj_data[obj_data['split'] == 'test'].iterrows():
            image_path = os.path.join(visa_root , row['image'])
            
            if row['label'] == 'normal':
                shutil.copy(image_path, normal_test_path)
                # shutil.copy(mask_path, normal_test_path)
            else:
                mask_path = os.path.join(visa_root, row["mask"])
                shutil.copy(image_path, anomaly_test_path)
                shutil.copy(mask_path, anomaly_mask_path)
        print(f"Object {object_name} processed!")

if __name__ == "__main__":
    visa_path = "root/visa"       
    target_path = "root/visa_mvtec"   
    
    convert_visa_to_mvtec(visa_path, target_path)
    print("Conversion completed!")