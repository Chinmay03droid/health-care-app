# Health Care Cost Prediction App 🚑💰

A simple web application that predicts healthcare insurance costs using machine learning (XGBoost). The app is containerized using Docker for easy deployment.

---

## 🔧 Features

- Predicts individual healthcare costs based on user inputs
- Uses an XGBoost regression model
- Preprocessing pipeline with custom logic
- Minimal HTML frontend using Flask
- Dockerized for consistent deployment

---

## 📦 Tech Stack

- Python 3
- Flask
- XGBoost
- Pandas, NumPy
- Scikit-learn
- HTML (Jinja templates)
- Docker

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Chinmay03droid/health-care-app.git
cd health-care-app


### 2. Run the docker image
docker build -t health-care-app .

### 3. Run the docker container
docker run -p 5000:5000 health-care-app

### 4. Accessing the App
Once the Docker container is running, open your browser and go to:
http://localhost:5000/
This will open the healthcare cost prediction app in your browser.

If you're running the app directly with Python (not Docker), use:
python health-care-app.py

## 📂 Project Structure
health-care-app/
│
├── Dockerfile
├── requirements.txt
├── health-care-app.py
├── xgboost_model.pkl
├── Preprocess_data.py
├── templates/
│   └── index.html
└── ...









