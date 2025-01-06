# Heart Attack Prediction Models

## Project Description
This repository contains multiple models developed for predicting heart attack risk using a dataset from [Kaggle](https://www.kaggle.com/datasets/mohamedmustafashaban/heart-attack-prediction). The results are presented in text files and visualizations from various machine learning models.

## Models Included
1. **RandomForestClassifier**
   - Confusion Matrix:
     ```
     [[1024  101]
      [ 580   48]]
     ```
   - Classification Report:
     ```
     precision    recall  f1-score   support

     0       0.64      0.91      0.75      1125
     1       0.32      0.08      0.12       628

     accuracy                           0.61      1753
     macro avg       0.48      0.49      0.44      1753
     weighted avg       0.53      0.61      0.53      1753
     ```

2. **XGBClassifier**
   - Confusion Matrix:
     ```
     [[881 244]
      [492 136]]
     ```
   - Classification Report:
     ```
     precision    recall  f1-score   support

     0       0.64      0.78      0.71      1125
     1       0.36      0.22      0.27       628

     accuracy                           0.58      1753
     macro avg       0.50      0.50      0.49      1753
     weighted avg       0.54      0.58      0.55      1753
     ```

3. **AdaBoostClassifier**
   - Confusion Matrix:
     ```
     [[841 284]
      [484 144]]
     ```
   - Classification Report:
     ```
     precision    recall  f1-score   support

     0       0.63      0.75      0.69      1125
     1       0.34      0.23      0.27       628

     accuracy                           0.56      1753
     macro avg       0.49      0.49      0.48      1753
     weighted avg       0.53      0.56      0.54      1753
     ```

4. **DecisionTreeClassifier**
   - Confusion Matrix:
     ```
     [[735 390]
      [367 261]]
     ```
   - Classification Report:
     ```
     precision    recall  f1-score   support

     0       0.67      0.65      0.66      1125
     1       0.40      0.42      0.41       628

     accuracy                           0.57      1753
     macro avg       0.53      0.53      0.53      1753
     weighted avg       0.57      0.57      0.57      1753
     ```

5. **Pegasos_qsvc_qiskit_heart_attack**
   - Model Accuracy on Training Data: 50.97%
   - Root Mean Squared Error (RMSE): 0.84
   - Confusion Matrix on Training Data:
     ```
     [[2454 2045]
      [1392 1119]]
     ```

6. **QuantumKernelTrainer**
   - Model Accuracy on Training Data: 65.80%
   - Root Mean Squared Error (RMSE): 0.76
   - Confusion Matrix on Training Data:
     ```
     [[438 203]
      [139 220]]
     ```

## Dataset Source
This project utilizes the dataset from [Kaggle](https://www.kaggle.com/datasets/mohamedmustafashaban/heart-attack-prediction).

## Comparison and Conclusion

### Comparison:
- **RandomForestClassifier** achieved the highest accuracy of 61%, but with a significant imbalance between precision and recall for both classes.
- **XGBClassifier** provided slightly lower accuracy at 58%, with better precision and recall for the positive class compared to RandomForest.
- **AdaBoostClassifier** had an accuracy of 56%, showing lower overall performance compared to RandomForest and XGB.
- **DecisionTreeClassifier** had the lowest accuracy of 57%, with moderate precision and recall values.
- **Pegasos_qsvc_qiskit** provided a relatively lower accuracy (50.97%) with a higher Root Mean Squared Error (RMSE) compared to other models.
- **QuantumKernelTrainer** achieved a higher accuracy of 65.80%, offering a balanced precision and recall for both classes.

### Conclusion:
From the results, the **QuantumKernelTrainer** and **RandomForestClassifier** stand out with the highest accuracies of 65.80% and 61%, respectively. However, QuantumKernelTrainer provides a more balanced performance across both classes. While other models like AdaBoost and DecisionTree performed moderately, they show room for improvement in both accuracy and class balance. The choice of model largely depends on the specific requirements and constraints of the project, including the desired balance between accuracy and computational efficiency.

