# Predictive Modeling Using Machine Learning

## Overview
This beginner-friendly supervised-learning project trains a Random Forest classifier to predict a synthetic student's pass/fail outcome from sample study and performance features.

## Dataset
`student_performance.csv` has 100 synthetic demonstration records:
- `study_hours_per_day`
- `attendance_percent`
- `previous_exam_score`
- `assignments_completed`
- `passed` (1 = passed, 0 = not passed)

These are artificially generated records, not real student data.

## What the project does
- Splits data into training and test sets
- Trains a Random Forest classifier
- Evaluates accuracy, classification report, confusion matrix, and ROC-AUC
- Creates confusion-matrix, ROC-curve, and feature-importance charts
- Saves test predictions and feature importance to CSV

## Tools
Python, Pandas, Scikit-learn, Matplotlib.

## How to run
1. Install Python 3.
2. Run `pip install -r requirements.txt`.
3. Run `python analysis.py`.

The script creates an `outputs/` folder containing the charts and result CSV files.

## Important limitation
This is an educational demonstration only. Do not use it to make real decisions about students. Real-world use requires representative data, privacy protections, bias checks, and careful validation.

## Files
- `analysis.py` — model training and evaluation
- `student_performance.csv` — synthetic sample data
- `requirements.txt` — dependencies
- `README.md` — documentation
