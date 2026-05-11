import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split


def train_random_forest(df, model_output_path="rf_model.joblib"):
    # Encode labels into numbers
    label_mapping = {'converging': 0, 'stable': 1, 'diverging': 2}
    df['label_encoded'] = df['gt_label'].map(label_mapping)

    # Define feature columns
    feature_cols = [
        'slope_dispersion',
        'avg_dispersion',
        'dispersion_std_dev',
        'dispersion_change',
        'segment_length_frames'
    ]

    X = df[feature_cols]
    y = df['label_encoded']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=None, shuffle=False
    )

    # Train Random Forest
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_mapping.keys()))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))

    # Save model
    joblib.dump(clf, model_output_path)
    print(f"Model saved to {model_output_path}")

    cm = confusion_matrix(y_test, y_pred)
    labels = ['Converging', 'Stable', 'Diverging']
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    importances = clf.feature_importances_
    features = X.columns
    plt.barh(features, importances)
    plt.autoscale(enable=True)
    plt.title("Feature Importances")
    plt.xlabel("Importance Score")
    plt.show()

    # Get the report as a dictionary
    report_dict = classification_report(y_test, y_pred, target_names=labels, output_dict=True)

    # Create a formatted table
    cell_text = []
    rows = []
    metrics = ['precision', 'recall', 'f1-score', 'support']

    for label in labels:
        rows.append(label)
        row_data = [f"{report_dict[label][metric]:.2f}" for metric in metrics]
        cell_text.append(row_data)

    # Add overall averages if needed
    rows.append('Accuracy')
    acc = accuracy_score(y_test, y_pred)
    cell_text.append(['', '', f"{acc:.2f}", f"{len(y_test)}"])

    # Plot as table
    plt.figure(figsize=(8, len(labels) * 0.6 + 2))
    plt.axis('off')
    plt.title("Classification Report", fontsize=14, fontweight='bold')
    table = plt.table(
        cellText=cell_text,
        rowLabels=rows,
        colLabels=metrics,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    plt.tight_layout()
    plt.show()

    return clf, label_mapping


def get_dataframe(dir_path):
    df = pd.DataFrame(columns=[
        "video", "segment_id", "start_frame", "end_frame", "slope_dispersion",
        "avg_dispersion", "dispersion_std_dev", "dispersion_change",
        "segment_length_frames", "pred_label", "gt_label"
    ], index=None)

    videos = os.listdir(dir_path)
    for video in videos:
        df = pd.concat([df, pd.read_csv(f"{dir_path}/{video}/{video}_segment_data.csv""")], ignore_index=True)

    return df


# Example usage:
if __name__ == "__main__":
    path = "path_to_segment_labels"  # Replace with actual path
    df = get_dataframe(path)
    trained_model, label_map = train_random_forest(df)
