# **Sentiment Analysis on Amazon Product Reviews**

## **Project Overview**
This project performs sentiment analysis on Amazon product reviews to classify them as either **positive** or **negative**. Several machine learning models were trained and evaluated using accuracy, precision, recall, and F1 score to determine the best-performing model.

## **Dataset Overview**
The dataset consists of Amazon product reviews, with the following columns:
- **`reviewText`**: The textual content of the review.
- **`Positive`**: A binary label (1 for positive, 0 for negative).

## **Steps in the Project**

### 1. **Data Preprocessing**
- **Text Preprocessing**: Lowercased the text, removed punctuation, and filtered out stop words.
- **Missing Values**: Removed any rows with missing data in the relevant columns.
- **Data Splitting**: The dataset was split into training and testing sets.

### 2. **Model Selection**
Three machine learning models were chosen for comparison:
- **Logistic Regression**
- **Random Forest**
- **Support Vector Machine (SVM)**

The review text was vectorized using **TF-IDF** to convert it into numerical form suitable for machine learning algorithms.

### 3. **Model Training**
The models were trained using the training data and their performance was evaluated on the test set.

### 4. **Model Evaluation**
The models were evaluated based on:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

### 5. **Hyperparameter Tuning**
Grid Search was employed for hyperparameter tuning, particularly for the Random Forest model, to improve performance.

## **Comparative Analysis**

The table below summarizes the performance of each model based on the evaluation metrics:

| Model               | Accuracy | Precision | Recall  | F1 Score |
|---------------------|----------|-----------|---------|----------|
| Logistic Regression  | 0.8925   | 0.8999    | 0.9661  | 0.9318   |
| Random Forest        | 0.8735   | 0.8822    | 0.9622  | 0.9204   |
| SVM                  | 0.8938   | 0.9030    | 0.9638  | 0.9324   |

### **Key Findings**
- **Logistic Regression**: Demonstrated excellent performance across all metrics, with a high **F1 score** of 0.9318, making it well-balanced for applications where both false positives and false negatives matter.
  
- **Random Forest**: While it has slightly lower accuracy and precision, it achieved a high recall of 0.9622, indicating that it identifies positive reviews well. It’s best suited for scenarios where recall is more critical than precision.

- **SVM**: Performed the best in terms of **precision** (0.9030) and slightly higher **accuracy** than Logistic Regression, making it ideal for use cases where minimizing false positives is crucial.

### **Conclusion**
- **Logistic Regression** is the best overall model with balanced performance across accuracy, precision, recall, and F1 score.
- **Random Forest** excels when **recall** is the priority, though it trades off precision slightly.
- **SVM** is the top choice when **precision** is the most important metric, as it minimizes false positives effectively.

## **Technologies Used**
- **Python**: Pandas, Scikit-learn, NLTK
- **TF-IDF Vectorization**
- **Machine Learning Models**: Logistic Regression, Random Forest, SVM
- **Hyperparameter Tuning**: Grid Search

