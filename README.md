# Real-Time Face & Hand Tracking App

![Project Banner](https://via.placeholder.com/1000x300?text=Real-Time+Face+%26+Hand+Tracking+App)

## 📌 Project Overview
This project is a **real-time face and hand tracking application** built using **Streamlit, OpenCV, and MediaPipe**. It allows users to detect and track facial landmarks and hand movements through a webcam. The project is modularized, with face and hand tracking implemented in separate modules, making it scalable and maintainable.

## 📂 Project Structure
```
/Hand-Face-Tracking-Application
│── /modules
│   ├── facetrackingModule.py  # Face tracking module
│   ├── handtrackingModule.py  # Hand tracking module
│── app.py               # Main Streamlit application
│── requirements.txt     # Python dependencies
│── README.md            # Project documentation
```

## 🔧 Features
✅ **Face Tracking:** Uses MediaPipe Face Mesh to detect facial landmarks.  
✅ **Hand Tracking:** Uses MediaPipe Hands to detect and track hand positions.  
✅ **Real-Time Processing:** Displays FPS for performance monitoring.  
✅ **Streamlit UI:** Provides an interactive user interface with toggles for enabling/disabling tracking.  
✅ **Modular Code:** Face and hand tracking are implemented in separate files under `modules/` for easy maintenance.  

## 🛠 Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/hand-face-tracking-application.git
cd hand-face-tracking-application
```

### 2️⃣ Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Application
```bash
streamlit run app.py
```

## 🚀 Usage Instructions
1. Open the **Streamlit app** in your browser after running `streamlit run app.py`.
2. Use the **sidebar checkboxes** to enable/disable face and hand tracking.
3. The **video stream** will display real-time tracking.
4. Click **Stop Stream** to stop the webcam feed.

## 📜 Modules Overview

### `modules/facetracking.py`
- Uses **MediaPipe Face Mesh** to detect facial landmarks.
- Draws contour lines for better visualization.

### `modules/handtracking.py`
- Uses **MediaPipe Hands** to detect and track hand movements.
- Draws connections between finger joints for clear visualization.

## 🛠 Troubleshooting
- **Issue: Camera Not Opening?**  
  ✅ Ensure your webcam is not being used by another application.  
  ✅ Restart the application and try again.  

- **Issue: Slow Performance?**  
  ✅ Reduce the resolution of the webcam feed in `cv2.VideoCapture()`.  
  ✅ Close unnecessary applications running in the background.  

## 📌 Future Improvements
🔹 Add gesture recognition for more interactive controls.  
🔹 Support multiple cameras for a more flexible setup.  
🔹 Implement object tracking for additional functionalities.  

## 📜 License
This project is licensed under the **MIT License**.

## 🤝 Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

## 📬 Contact
For any inquiries or suggestions, reach out to **[Muhammad Inaamullah](mailto:muhammad.inaamullah@outlook.com)**.

---
💡 **Star this repository if you found it useful!** ⭐

