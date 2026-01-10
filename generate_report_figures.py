import os
import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def main() -> None:
    warnings.filterwarnings("ignore")

    repo_root = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(repo_root, "content", "Student_performance_data _.csv")
    out_dir = os.path.join(repo_root, "assets")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    df.drop_duplicates(inplace=True)

    # Round as in notebook
    if "GPA" in df.columns:
        df["GPA"] = df["GPA"].round(2)
    if "StudyTimeWeekly" in df.columns:
        df["StudyTimeWeekly"] = df["StudyTimeWeekly"].round(2)

    # Create Results target
    df["Results"] = np.where(df["GradeClass"] <= 1.5, "Fail", "Pass")

    # Encode categorical columns (matching notebook intent)
    cat_columns = ["Gender", "Ethnicity", "ParentalEducation"]
    le = LabelEncoder()
    for col in cat_columns:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])

    # Scale numeric columns (matching notebook)
    numeric_features = ["Age", "StudyTimeWeekly", "Absences", "GPA"]
    scaler = StandardScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    # --- Figure 1: Class distribution ---
    plt.figure(figsize=(6, 4))
    order = ["Pass", "Fail"]
    sns.countplot(x="Results", data=df, order=order)
    plt.title("Class Distribution (Pass vs Fail)")
    plt.xlabel("Results")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "class_distribution.png"), dpi=200)
    plt.close()

    # --- Figure 2: GPA distribution (scaled) ---
    plt.figure(figsize=(10, 6))
    sns.histplot(df["GPA"], kde=True, bins=20)
    plt.title("GPA Distribution")
    plt.xlabel("GPA")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gpa_distribution.png"), dpi=200)
    plt.close()

    # --- Figure 3: StudyTimeWeekly vs GPA (scaled) ---
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="StudyTimeWeekly", y="GPA", s=18)
    plt.title("Study Time vs GPA")
    plt.xlabel("Study Time Weekly")
    plt.ylabel("GPA")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "study_time_vs_gpa.png"), dpi=200)
    plt.close()

    # --- Train model to reproduce confusion matrix + feature importance ---
    X = df.drop(columns=["Results", "StudentID", "GradeClass"])
    y = df["Results"]
    y = LabelEncoder().fit_transform(y)  # Fail=0, Pass=1

    # NOTE: Notebook did not use stratify; keep same to match earlier outputs.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [5, 10, 15],
        "max_features": ["sqrt", "log2"],
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring="accuracy")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # --- Figure 4: Confusion matrix ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Fail", "Pass"],
        yticklabels=["Fail", "Pass"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=200)
    plt.close()

    # --- Figure 5: Feature importances ---
    importances = best_model.feature_importances_
    importance_df = (
        pd.DataFrame({"Feature": X.columns, "Importance": importances})
        .sort_values(by="Importance", ascending=False)
        .reset_index(drop=True)
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(data=importance_df, x="Importance", y="Feature")
    plt.title("Feature Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "feature_importances.png"), dpi=200)
    plt.close()

    # Also save the numbers used for the report
    importance_df.to_csv(os.path.join(out_dir, "feature_importances.csv"), index=False)

    print("Saved figures to:", out_dir)
    print("Saved feature importances CSV to:", os.path.join(out_dir, "feature_importances.csv"))


if __name__ == "__main__":
    main()
