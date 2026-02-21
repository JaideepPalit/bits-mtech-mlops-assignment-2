
from pathlib import Path
import subprocess
import sys
import kagglehub
import os
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def download_dataset(url:str,download_path:str):
    download_path = Path(__file__).resolve().parents[2] / "data"/"raw"/download_path
    path = kagglehub.dataset_download(handle=url,output_dir=str(download_path))
    print("Path to dataset files:", path)
    data_dir = os.path.join(path, 'PetImages')
    return data_dir


def pre_process_dataset(data_dir:str,output_dir:str="preprocessed_cats_dogs_images"):
    output_dir = Path(__file__).resolve().parents[2] / "data"/ "preprocessed"/output_dir
    os.makedirs(output_dir,exist_ok=True)
    categories = ['Cat', 'Dog']
    splits = ['train', 'val', 'test']

    for s in splits:
        for cat in categories:
            os.makedirs(os.path.join(output_dir, s, cat), exist_ok=True)

    def process_and_save(file_list, category, split_name, target_size=(224, 224)):
        """Resizes, converts to RGB, and saves images to the new directory."""
        for file_path in tqdm(file_list, desc=f"Processing {category} for {split_name}"):
            try:
                # Read image
                img = cv2.imread(file_path)
                if img is None: continue
                
                # Preprocess: Resize to 224x224
                img_resized = cv2.resize(img, target_size)
                
                # Save to new location
                file_name = os.path.basename(file_path)
                save_path = os.path.join(output_dir, split_name, category, file_name)
                cv2.imwrite(save_path, img_resized)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    # 3. Perform the Split and Execute
    for category in categories:
        cat_folder = os.path.join(data_dir, category)
        all_files = [os.path.join(cat_folder, f) for f in os.listdir(cat_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Split 80% Train, 20% Temp (Val + Test)
        train_files, temp_files = train_test_split(all_files, test_size=0.20, random_state=42)
        
        # Split the 20% Temp into 50/50 (which is 10% / 10% of total)
        val_files, test_files = train_test_split(temp_files, test_size=0.50, random_state=42)
        
        # Process and write to disk
        process_and_save(train_files, category, 'train')
        process_and_save(val_files, category, 'val')
        process_and_save(test_files, category, 'test')

    print(f"\n✅ Dataset saved successfully at: {os.path.abspath(output_dir)}")


def run_cmd(cmd):
    """Run shell command safely and exit on failure."""
    print(f"\nRunning: {cmd}")
    result = subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        print("ERROR:")
        print(result.stderr)
        sys.exit(1)

    print(result.stdout)

def data_versioning_with_dvc():
    

    run_cmd("dvc remote add -f origin s3://dvc")
    run_cmd("dvc remote modify origin endpointurl https://dagshub.com/jaideep.palit/bits-mtech-mlops-assignment-2.s3")
    
    run_cmd("dvc remote modify origin --local access_key_id d85f991495a6411e956277b0781bd119dfac225d")
    run_cmd("dvc remote modify origin --local secret_access_key d85f991495a6411e956277b0781bd119dfac225d")

    # -------------------------------------------------
    # 4. Track dataset
    # -------------------------------------------------
    processed_file_path = Path(__file__).resolve().parents[2] / "data"/ "preprocessed"/"preprocessed_cats_dogs_images"

    run_cmd(f"dvc add {processed_file_path}")

    # -------------------------------------------------
    # 5. Push data to DagsHub
    # -------------------------------------------------
    run_cmd("dvc push -r origin")

    print("\nDVC setup completed successfully")

def git_dvc_version():
    processed_file_path_dvc = Path(__file__).resolve().parents[2] / "data"/ "preprocessed"/"preprocessed_cats_dogs_images.dvc"
    run_cmd("git remote set-url origin https://JaideepPalit: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@github.com/JaideepPalit/bits-mtech-mlops-assignment-2.git")
    # Add specific file
    run_cmd(f"git add {processed_file_path_dvc}")

    # Commit (safe if nothing changed)
    run_cmd(f"git commit -m 'Track {processed_file_path_dvc} using DVC' || echo 'Nothing to commit'")

    # Push to remote
    run_cmd("git push origin main")

    print("✅ File pushed to Git successfully")
    pass