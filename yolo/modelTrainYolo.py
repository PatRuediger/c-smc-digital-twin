
from ultralytics import YOLO
import os

if __name__ == "__main__":
    # Define paths
    base_dir = '/scratch/ruediger/Exp_05_Vorversuche_YOLO/'
    output_dataset_dir = os.path.join(base_dir, 'runs/detect_11s')
    yaml_path = os.path.join(base_dir, 'dataset/data.yaml')

    # Load a pre-trained YOLOv11s (small) model
    model = YOLO('yolo11s.pt')

    # Define project and run name for storing results
    # The results will be saved to project/name
    project_dir = os.path.join(base_dir, 'runs/detect_11s')
    run_name = 'train'

    # Train the model using your YAML file
    results = model.train(data=yaml_path, epochs=100, imgsz=512, project=project_dir, name=run_name, batch=16)

    # The results object contains the path to the directory where results are saved
    # We can use this path in other cells
    results_dir = results.save_dir
    print(f"Results saved to: {results_dir}")