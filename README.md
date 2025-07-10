

# 🧠 Stress Detection Using Encoded Physiological Signals

This project implements a robust, explainable AI pipeline to detect human stress using deep learning models on encoded physiological signal representations. It leverages the **NEURO** and **WESAD** datasets, applies **image encoding techniques** like **GAF, GAFd, and MTF**, and uses **CNNs** with **Grad-CAM** for interpretability.

---

## 📊 Datasets Used

* **NEURO Dataset**: Includes BVP, EDA, temperature, HR, and motion data collected during physical, cognitive, and emotional stress tasks.
* **WESAD Dataset**: Multimodal data from wrist and chest sensors capturing EDA, ECG, EMG, respiration, temperature, and acceleration.

---

## ⚙️ Methodology

### 1. **Data Preprocessing**

* Used sliding window techniques along **time** and **feature** axes.
* Normalized physiological signals using `MinMaxScaler` and `StandardScaler`.

### 2. **Signal-to-Image Transformation**

* **GAF (Gramian Angular Field)**
* **GAFd (Derivative GAF)**
* **MTF (Markov Transition Field)**

Converted time-series windows into 2D images of shape 64×64×7 (or 105×105 in WESAD).

### 3. **Model Architectures**

* **Classical ML Models**: XGBoost, LightGBM, CatBoost, Random Forest, etc.
* **Deep Learning**:

  * Raw signal-based: 1D-CNN, LSTM, CNN+LSTM
  * Image-based: CNN on GAF/GAFd/MTF images

### 4. **Explainability**

* **Grad-CAM** used to visualize important regions in signal-encoded images influencing CNN decisions.

---

## 📈 Evaluation

* **Metrics**: Accuracy, Precision, Recall, F1-score, Specificity
* **Validation**:

  * **Leave-One-Subject-Out (LOSO)**
  * **Cross-Validation (CV)**

> Best performance:
>
> * **MTF + CNN (CV)** → 99.57% Accuracy
> * **GAFd + CNN (CV)** → Fastest convergence
> * **Random Forest (LOSO)** → Best classical model (77.6% Accuracy)

---

## 🔍 Results Summary

| Encoding | Method | Accuracy (LOSO) | Accuracy (CV) |
| -------- | ------ | --------------- | ------------- |
| GAF      | CNN    | 69.96%          | 98.36%        |
| GAFd     | CNN    | 77.89%          | 98.35%        |
| MTF      | CNN    | **79.97%**      | **99.57%**    |

Grad-CAM analysis shows models rely on sharp angular transitions in physiological signal images for prediction.

---

## 🧠 Key Takeaways

* Image-based representations improve generalization across subjects.
* Feature-axis sliding + GAFd/MTF encoding yields highest accuracy.
* Grad-CAM makes CNN predictions interpretable and trustworthy.

---

## 🛠 Tech Stack

* Python (PyTorch, Scikit-learn)
* `pyts` for time-series image conversion
* Jupyter Notebooks
* Matplotlib/Seaborn for visualization

---

## 📚 Future Work

* Integrate contextual data (activity, location)
* Explore semi-supervised and personalized learning
* Use SHAP/LIME for model-agnostic explanations


