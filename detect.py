from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    model = YOLO(model=r"runs/train/result_1_95data/weights/best.pt")
    model.predict(
        source=r"datasets/dataset_1/images/val/0624162353_57.jpg",
        save=True,
        show=True,
    )
