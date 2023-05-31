# Project Report: StarCraft Player Analysis

## Dataset Description

The dataset consisted of various features related to StarCraft players, including demographic data, gameplay features, and the corresponding skill level (LeagueIndex).

## Exploratory Data Analysis (EDA)

During the EDA phase, I visualized the distribution of players across different skill levels using a histogram. The histogram revealed an imbalanced dataset, with a higher number of players in lower skill levels compared to higher skill levels.

## Feature Importance

To understand the significance of each feature in predicting player skill level, I employed a RandomForestClassifier model and extracted the feature importances. The top five features contributing to the model's predictions were as follows:

1. ActionLatency: Importance - 17.29%
2. APM: Importance - 15.87%
3. SelectByHotkeys: Importance - 9.72%
4. NumberOfPACs: Importance - 9.68%
5. TotalHours: Importance - 8.32%

![Alt text](/images/feature.png?raw=true "Optional Title")


## Model Building and Evaluation

For model building, I used the RandomForestClassifier algorithm. After hyperparameter tuning using GridSearchCV, the best model achieved a mean cross-validation accuracy of 41.57%.

I evaluated the model's performance on the test set and obtained a test accuracy of 37.33% before calibration. Applying calibration techniques, specifically Platt Scaling, improved the test accuracy to 40.32%.

![Alt text](/images/accuracy.png?raw=true "Optional Title")

![Alt text](/images/graph.png?raw=true "Optional Title")

## Model Calibration

Calibration aimed to refine the predicted probabilities of the model. I applied Platt Scaling using the `CalibratedClassifierCV` class, resulting in calibrated predictions that were more aligned with the true probabilities.

After calibration, I assessed the percentage of predictions that fell within ±1 of the true skill level. Before calibration, approximately 86.63% of the predictions were within this range. After calibration, the percentage slightly decreased to 86.23%.

## Conclusion

In conclusion, this project analyzed the performance of StarCraft players and developed a predictive model for classifying their skill levels. Key features such as ActionLatency, APM, SelectByHotkeys, NumberOfPACs, and TotalHours played significant roles in predicting skill levels.

The RandomForestClassifier model achieved a test accuracy of 37.33%, which improved to 40.32% after calibration. The feature importances provide insights into the gameplay characteristics influencing skill levels.

The project highlights the importance of calibration in refining model predictions and offers valuable insights for evaluating player performance and improving skill levels in StarCraft.

Please note that the dataset exhibited class imbalance, and future work may involve addressing this imbalance to further enhance the model's performance.
