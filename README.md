
# 🚘 Number Plate Detection of Vehicles using YOLO & Streamlit

This project implements a **Number Plate Detection System** using **YOLO (Ultralytics)** for object detection and **EasyOCR** for optical character recognition (OCR).  
The system is deployed as an interactive **Streamlit web application** with support for **image upload**, **live camera detection**, **result logging**, and **performance analytics**.

---

## 📌 Project Objectives

- Detect vehicle number plates using deep learning (YOLO)
- Extract license plate text using OCR
- Provide a user-friendly GUI using Streamlit
- Support cloud deployment (Streamlit Community Cloud)
- Log detection results and analyze performance

---

## 🧠 Technologies Used

- **Python 3.10**
- **YOLO (Ultralytics)**
- **EasyOCR**
- **OpenCV (Headless)**
- **Streamlit**
- **Streamlit-WebRTC** (Live Camera)
- **Pandas / NumPy**
- **Matplotlib**

---

## 🗂️ Project Structure

```

number-plate-detection-of-vehicles/
│
├── NewStreamlit/
│   ├── Layout.py                  # Main layout & navigation
│   ├── ImageDetection.py          # Image upload detection
│   ├── LiveCameraDetection.py     # Live camera detection (WebRTC)
│   ├── Results.py                 # Detection logs & reports
│   ├── Dashboard.py               # Performance dashboard
│
├── runs/
│   └── exp/
│       └── weights/
│           └── best.pt             # Trained YOLO model
│
├── data/
│   └── detection_log.csv           # Detection log file
│
├── requirements.txt                # Python dependencies
├── packages.txt                    # System dependencies (Streamlit Cloud)
├── runtime.txt                     # Python version (3.10)
├── README.md                       # Project documentation

````

---

## 🚀 Features

### ✅ Image Detection
- Upload vehicle images (JPG / PNG)
- Detect number plates
- Display bounding boxes with confidence score
- OCR text shown above bounding box

### ✅ Live Camera Detection
- Real-time number plate detection
- Browser-based webcam access (WebRTC)
- OCR applied on detected plates
- Works on **Streamlit Cloud & local machine**

### ✅ Results & Logs
- Stores:
  - Timestamp
  - Confidence threshold
  - OCR output
  - Image paths
- Export results as CSV
- View thumbnails of original and predicted images

### ✅ Performance Dashboard
- Total detections
- Average confidence score
- Detection trends over time
- OCR frequency analysis

---

## ⚙️ Installation (Local)

### 1️⃣ Clone Repository
```bash
git clone <your-github-repo-url>
cd number-plate-detection-of-vehicles
````

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate    # Linux / Mac
venv\Scripts\activate       # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Application

```bash
streamlit run NewStreamlit/Layout.py
```

---

## ☁️ Deployment (Streamlit Cloud)

This project is successfully deployed on **Streamlit Community Cloud**.

### Required Files for Deployment

* `requirements.txt`
* `packages.txt`
* `runtime.txt`

#### `runtime.txt`

```
python-3.10
```

#### `packages.txt`

```
libgl1
```

### Deployment Steps

1. Push project to GitHub (public repo)
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Select repository
4. Set entry point:

   ```
   NewStreamlit/Layout.py
   ```
5. Deploy 🚀

---

## 📦 Model Details

* **Model:** YOLO (Ultralytics)
* **Task:** Object Detection
* **Class:** `number_plate`
* **Model Path:**

  ```
  runs/exp/weights/best.pt
  ```

The model is loaded dynamically during inference.

---

## 📄 Submission Details (GUI Academy / MDU)

As per the project PDF instructions:

### ✔ Submitted Artifacts

* ✅ GitHub Repository (Code + README)
* ✅ Streamlit Web Application
* ✅ Performance Dashboard
* ✅ Detection Logs (CSV)
* ✅ Model Weights
* ✅ GUI-based Output

### ✔ Submission Method

* **GitHub Repository Link**
* (Optional) Deployed Streamlit App URL

---

## 🧪 Tested Environments

| Environment     | Status                       |
| --------------- | ---------------------------- |
| Local Machine   | ✅ Working                    |
| Streamlit Cloud | ✅ Working                    |
| Google Colab    | ⚠️ Live camera not supported |

---

## 📚 References

* Streamlit Documentation
  [https://docs.streamlit.io/](https://docs.streamlit.io/)

* Ultralytics YOLO
  [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

* EasyOCR
  [https://github.com/JaidedAI/EasyOCR](https://github.com/JaidedAI/EasyOCR)

* Streamlit WebRTC
  [https://github.com/whitphx/streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc)

---

## 👤 Author

**Name:** Sakthi Gnana Prakasam
**Project:** Number Plate Detection of Vehicles
**Institute:** GUI Academy / MDU

---

## 🏁 Conclusion

This project demonstrates a complete **end-to-end computer vision system**, integrating deep learning, OCR, and web deployment.
It satisfies all requirements mentioned in the **Streamlit Integration** and **Performance Dashboard** sections of the project guidelines.

---


