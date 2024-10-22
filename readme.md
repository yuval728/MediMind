# Medicine Recommendation System

This project is a Medicine Recommendation System built using a Decision Tree Classifier. The system predicts diseases based on symptoms and provides recommendations for medications, precautions, workouts, and diets.

## Features

- **Disease Prediction**: Predicts the disease based on user-selected symptoms.
- **Recommendations**: Provides detailed recommendations including description, precautions, workouts, medications, and diets for the predicted disease.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yuval728/MediMind.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Medicine-recommendation
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Ensure the model and label encoder files are in the `artifacts/` directory.
2. Ensure the dataset files are in the `dataset/` directory.
3. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

## Files

- `artifacts/DecisionTreeClassifier.pkl`: Trained Decision Tree Classifier model.
- `artifacts/label_encoder.pkl`: Label encoder for disease labels.
- `dataset/symtoms_df.csv`: Dataset containing symptoms.
- `dataset/precautions_df.csv`: Dataset containing precautions for diseases.
- `dataset/workout_df.csv`: Dataset containing workout recommendations for diseases.
- `dataset/description.csv`: Dataset containing descriptions of diseases.
- `dataset/medications.csv`: Dataset containing medication recommendations for diseases.
- `dataset/diets.csv`: Dataset containing diet recommendations for diseases.

## How It Works

1. **Load Model and Data**: The app loads the trained model, label encoder, and datasets.
2. **User Input**: Users select symptoms from a list.
3. **Prediction**: The app predicts the disease based on the selected symptoms.
4. **Recommendations**: The app provides detailed recommendations for the predicted disease.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Joblib](https://joblib.readthedocs.io/en/latest/)
