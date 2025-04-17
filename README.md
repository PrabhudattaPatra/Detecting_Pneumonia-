# Pneumonia Detection from Chest X-Ray

This project is a web-based application built using **Streamlit** and **TensorFlow** that detects whether a chest X-ray image shows signs of **Pneumonia** or is **Normal**.

##  Demo

Upload a chest X-ray 4D image, and the app will analyze it using a trained Convolutional Neural Network (CNN) to predict:
- **NORMAL**
- **PNEUMONIA**

## Model

- Trained on a dataset of labeled chest X-ray images (sourced from Kaggle).
- Used a custom Convolutional Neural Network (CNN) model.
- Model trained and evaluated on separate train, validation, and test sets.
- Final model saved as `best_model.h5`.

###  Model Performance

![image](https://github.com/user-attachments/assets/2a1fc1d2-ba16-4976-a318-26fdb4139ccd)


> The model prioritizes high **recall** for Pneumonia (important in medical diagnosis), even at the cost of some false positives.

## Technologies Used

- Python 
- TensorFlow 
- Streamlit 
- Matplotlib & Seaborn
- Scikit-learn (for evaluation metrics)
- PIL / NumPy

## Project Structure

```
├── chest_xray/              # Dataset (train/val/test folders)
├── streamlit_app.py         # Main Streamlit interface
├── model_training.ipynb     # Jupyter notebook for training & evaluation
├── best_model.h5            # Saved CNN model            
├── README.md                # Project documentation
```

## Sample Images

You can test the app using chest X-ray images of:
- Healthy lungs (**NORMAL**)
- Infected lungs (**PNEUMONIA**)

Sample predictions and error analysis (false positives/negatives) are also visualized in the notebook.

##  Setup Instructions

1. **Clone the repo**
```bash
git clone https://github.com/PrabhudattaPatra/Detecting_Pneumonia-.git
cd Detecting_Pneumonia-
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app**
```bash
streamlit run streamlit_app.py
```

## Results & Analysis

- Confusion Matrix, Classification Report, and misclassified image samples provided.
- High Pneumonia recall makes this model useful for screening purposes.
- Error analysis revealed the model sometimes mislabels clear NORMAL cases as PNEUMONIA.

## Future Improvements

- Improve NORMAL class recall.
- Experiment with transfer learning (e.g., EfficientNet).
- Deploy as a web service or mobile app.



