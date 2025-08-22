# Heart Disease Prediction

This project implements a **machine learning model** for predicting the likelihood of heart disease based on medical attributes.  
It applies data preprocessing, model training, and evaluation to provide insights into risk classification.

## Features
- Data preprocessing and cleaning (handling missing values, normalization, etc.)
- Exploratory Data Analysis (EDA) with visualizations
- Training machine learning models (Logistic Regression, Decision Tree, Random Forest, etc.)
- Model evaluation using accuracy, precision, recall, and F1-score
- User-friendly interface for making predictions

## Tech Stack
- **Python**
- **Pandas**, **NumPy** – data handling
- **Matplotlib**, **Seaborn** – visualization
- **Scikit-learn** – ML models and evaluation

## Dependencies
Make sure you have the following Python packages installed:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

You can install them using:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Project Structure
heart-disease-prediction/
│
├── data/ # Dataset 

├── src/ # Python scripts for preprocessing and training

└── README.md # Project documentation

---

## How to Run

Follow these steps to run the Heart Disease Prediction project:

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/heart-disease-prediction.git
cd heart-disease-prediction 
```

### 2. Install Dependencies

Make sure you have Python installed.
Then install required packages:

pip install pandas numpy scikit-learn matplotlib seaborn

### 3. Run the Model

You can run the training script:

python src/train.py


### 4. Dataset
The project uses the UCI Heart Disease Dataset.

## Results

Achieved accuracy of 82%.

Random Forest showed the best performance among tested models.

## License

This project is licensed under the MIT License.
