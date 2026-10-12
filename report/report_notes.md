# Report notes 

The course brief asks for a 4–6 page research-style report with a maximum of 2 images.

## 1. Introduction
Explain the binary classification problem and why Fiction vs Nonfiction is interesting.

## 2. Data Scraping
Describe:
- source website
- how category pages were identified
- fields scraped
- how pagination was handled
- image download strategy
- request/session considerations

## 3. Data Cleaning and Analysis
Document:
- duplicate handling
- missing values
- numeric conversion
- class distribution
- title statistics
- any records removed

## 4. Feature Engineering
Explain why you selected:
- TF-IDF title features
- price
- rating
- title length
- cover color statistics

Also record experiments where a feature was removed and performance changed.

## 5. Model Training
Explain the chosen classifier and why.

A useful comparison is:
- Logistic Regression
- Random Forest

Do not claim a result until you have actually run the experiment.

## 6. Model Evaluation
Report:
- holdout accuracy
- precision
- recall
- F1
- 5-fold cross-validation
- confusion matrix

Discuss false positives and false negatives.

## 7. Conclusion and Discussion
Discuss:
- whether the model learned useful patterns
- limitations of the dataset
- possible data leakage or weak proxies
- small/limited class coverage
- what you would improve with more time

## 8. References
Cite the source website and the libraries/model methodology you actually used.
