# 🌿 Coriander vs Parsley Classifier API

This API is designed to classify images as either **Coriander** or **Parsley** using a machine learning model built with PyTorch and FastAPI. It leverages a DenseNet-201 architecture, fine-tuned for this binary classification task.

## 🚀 Features

- Accepts image uploads for classification.
- Returns the predicted class: **Coriander** or **Parsley**.
- Includes robust image preprocessing with transformations like resizing, cropping, and normalization.
- Lightweight and efficient API with a pre-trained PyTorch model.

## 🛠️ Tech Stack

- **FastAPI**: For creating the RESTful API.
- **PyTorch**: For the machine learning model.
- **TorchVision**: For image transformation and model utilities.
- **Pillow (PIL)**: For image processing.
- **Render**: For seamless deployment.

## 📋 Endpoints

### `GET /`
- **Description**: A welcome endpoint to test if the API is running.
- **Response**: 
  ```json
  {
    "message": "Welcome to Coriander vs Parsley API!"
  }
  ```

### `POST /predict/`
- **Description**: Upload an image for classification.
- **Request**: 
  - Accepts a `file` parameter in `multipart/form-data` format.
  - Image formats: JPG, PNG, etc.
- **Response**: 
  - On success:
    ```json
    {
      "predicted_class": "Coriander"
    }
    ```
  - On error:
    ```json
    {
      "error": "Invalid image file. Error: ..."
    }
    ```

## ⚙️ Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the API locally**:
   ```bash
   uvicorn Coriander_vs_Parsley:app --reload
   ```

4. **Access the API**:
   - Visit: [http://127.0.0.1:8000](http://127.0.0.1:8000)

## 📡 Deployment

The API is deployed on **Render** for public access. You can use the following base URL for testing:
```
https://coriander-vs-parsley-api.onrender.com
```

## 📝 Notes

- Ensure the file `coriander_vs_parsley_model_weights.pth` is present in the root directory for successful model loading.
- Model weights are currently mapped for CPU execution. Update device configurations as needed for GPU acceleration.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to fork the repository and submit a pull request.
