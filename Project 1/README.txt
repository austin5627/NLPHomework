Running:
    python3 main.py

Requires:
    dataset folder with structure:
        main.py
        dataset/
            SemEval2010_task8_training/
                TRAIN_FILE.TXT
            SemEval2010_task8_testing_keys/
                TEST_FILE_FULL.TXT

output:
    training data statistics
    test data statistics
    model summary
    model training history
    model evaluation
    confusion matrix

incorrect_predictions.csv
    columns:
        sentence label predicted_label

stats.txt:
    training data statistics
    test data statistics
    accuracy and loss
    confusion matrix
