# Multi-Model Traffic Analysis  
This project leverages transfer learning and deep learning architectures to perform vehicle detection, traffic segmentation, and congestion prediction. Using state-of-the-art models like YOLOv8, Vision Transformer (ViT), and Attention U-Net, it provides a comprehensive pipeline for traffic analysis.

---

## Features  
- **Object Detection:** YOLOv8 detects vehicles and counts them in images.  
- **Segmentation:** Attention U-Net generates segmentation masks to identify regions of interest using an attention mechanism.  
- **Classification:** Vision Transformer (ViT) predicts traffic congestion based on vehicle count and global image features.  
- **Integration:** Combines all models to detect traffic jams and visualize results.  

---

## Prerequisites  

1. **Python Libraries:**  
   Install the required libraries:  
   ```bash
   pip install transformers segmentation-models-pytorch ultralytics roboflow
   ```

2. **Dataset Access:**  
   - Obtain the dataset using [Roboflow](https://roboflow.com/).  
   - Configure an API key to download the dataset.  

3. **Environment Setup:**  
   - GPU recommended for training and inference.  
   - Tested on PyTorch with CUDA support.  

---

## Dataset Structure  
The dataset should follow the YOLOv8 format, including:  
- **train/images/**: Training images.  
- **valid/images/**: Validation images.  
- **data.yaml**: Configuration file specifying dataset classes and paths.  

---

## Pipeline Overview  

### 1. **Vehicle Detection (YOLOv8):**  
   Detect vehicles in images and count them. YOLOv8 is trained on the dataset for optimal performance.  

### 2. **Image Segmentation (Attention U-Net):**  
   Generate segmentation masks to highlight regions of interest using an Attention U-Net with a VGG16 encoder.  

### 3. **Congestion Prediction (ViT):**  
   Classify images as "Traffic Jam" or "No Traffic Jam" using ViT with transfer learning.  

### 4. **Visualization:**  
   Results are visualized with bounding boxes, segmentation masks, and traffic predictions.  

---

## Code Example  

### Load and Test Models:  
```python
vit_model = load_vit_model()  
unet_model = load_attention_unet()  
yolo_model = YOLO("path/to/weights/best.pt").to(device)  

output_file = "results.txt"  
test_on_dataset('/path/to/validation/images', vit_model, unet_model, yolo_model, output_file)  
```

### Analyze an Image:  
```python
is_jam, vehicle_count = process_image(
    '/path/to/image.jpg',
    vit_model, unet_model, yolo_model
)  
print(f"Traffic Jam Detected: {is_jam}, Vehicle Count: {vehicle_count}")  
```

---

## Model Training  

### YOLOv8 Training:  
```bash
yolo_model.train(data="data.yaml", epochs=5, imgsz=640, batch=32, name="vehicle_detection", pretrained=True)
```

### Attention U-Net Fine-Tuning:  
Utilizes a pre-trained VGG16 encoder with an attention mechanism for segmentation.  

---

## Results  

- **Detections:** Accurate bounding boxes for vehicle identification.  
- **Segmentation Masks:** Precise masks for region segmentation with attention mechanisms.  
- **Traffic Jam Predictions:** Congestion classified with high accuracy.  

---

## Outputs  

1. Visualized Images:  
   - Bounding boxes with color-coded traffic status.  
2. Text Report:  
   - Logs traffic jam detections and vehicle counts in `results.txt`.  

---

## Contributing  

Contributions are welcome! Please fork the repository and submit a pull request.  

---

## Acknowledgments  
- [Hugging Face Transformers](https://huggingface.co/transformers/)  
- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)  
- [Ultralytics YOLO](https://ultralytics.com/)  
- [Roboflow API](https://roboflow.com/) 
