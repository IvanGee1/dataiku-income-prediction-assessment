import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from preprocessing import run_preprocessing
from feature_engineering import feature_engineering, map_income_target, get_grouping_values




def prepare_xy(df, target_col="income_binary"):

    df = df.copy()

    y = df[target_col]

    drop_cols = [target_col, "income"]

    if "instance weight" in df.columns:
        drop_cols.append("instance weight")

    X = df.drop(columns=drop_cols)

    return X, y



def build_logistic_pipeline(X):
   
    categorical_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    return pipeline


def build_random_forest_pipeline(X):
   
    categorical_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    return pipeline


def evaluate_setup(train_df, test_df, label, pipeline_builder):
 
    X_train, y_train = prepare_xy(train_df)
    X_test, y_test = prepare_xy(test_df)

    pipeline = pipeline_builder(X_train)
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)

    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        roc_auc = None

    categorical_cols = X_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    transformed_feature_names = numeric_cols + categorical_cols

    results = {
        "setup": label,
        "n_input_features": X_train.shape[1],
        "feature_names": transformed_feature_names,
        "pipeline": pipeline,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc,
        "report": classification_report(y_test, preds, digits=4, zero_division=0),
    }

    return results


def print_results(results):
   
    print(f"{results['setup']}")
    print(f"Number of input features: {results['n_input_features']}")
    print(f"Accuracy : {results['accuracy']}")
    print(f"Precision: {results['precision']}")
    print(f"Recall   : {results['recall']}")
    print(f"F1 Score : {results['f1']}")

    if results["roc_auc"] is not None:
        print(f"ROC-AUC  : {results['roc_auc']}")

    print("\nClassification report:")
    print(results["report"])



def main():
 
    learn_clean, test_clean = run_preprocessing()

    # 1. Processed only
    train_base = map_income_target(learn_clean)
    test_base = map_income_target(test_clean)

    # 2. Learn grouping rules from train only
    learn_temp = map_income_target(learn_clean)

    grouping_rules = {
        "detailed household and family stat": get_grouping_values(
            learn_temp, "detailed household and family stat"
        ),
        "family members under 18": get_grouping_values(
            learn_temp, "family members under 18"
        ),
    }

    # 3. Processed + engineered
    train_fe = feature_engineering(learn_clean, grouping_rules=grouping_rules)
    test_fe = feature_engineering(test_clean, grouping_rules=grouping_rules)

    # Logistic regression: baseline vs feature engineered
    base_results = evaluate_setup(
        train_base,
        test_base,
        "Logistic Regression - processed data",
        build_logistic_pipeline,
    )

    fe_log_results = evaluate_setup(
        train_fe,
        test_fe,
        "Logistic Regression - processed and feature engineered",
        build_logistic_pipeline,
    )

    # Random forest on the feature engineered version
    fe_rf_results = evaluate_setup(
        train_fe,
        test_fe,
        "Random Forest - processed data and feature engineered",
        build_random_forest_pipeline,
    )

    print_results(base_results)
    print_results(fe_log_results)
    print_results(fe_rf_results)



    print("\nFeature engineering impact for Logistic Regression:")

    print(f"- Input features: {base_results['n_input_features']} -> {fe_log_results['n_input_features']}")

    acc_diff = fe_log_results['accuracy'] - base_results['accuracy']
    prec_diff = fe_log_results['precision'] - base_results['precision']
    rec_diff = fe_log_results['recall'] - base_results['recall']
    f1_diff = fe_log_results['f1'] - base_results['f1']
    roc_diff = fe_log_results['roc_auc'] - base_results['roc_auc']

    print(f"- Accuracy change : {acc_diff}")
    print(f"- Precision change: {prec_diff}")
    print(f"- Recall change   : {rec_diff}")
    print(f"- F1 change       : {f1_diff}")
    print(f"- ROC-AUC change  : {roc_diff}")


    print("\nModel comparison on engineered features:")

    print(f"- Logistic Regression F1: {fe_log_results['f1']}")
    print(f"- Random Forest F1     : {fe_rf_results['f1']}")

    print(f"- Accuracy diff : {fe_rf_results['accuracy'] - fe_log_results['accuracy']}")
    print(f"- Precision diff: {fe_rf_results['precision'] - fe_log_results['precision']}")
    print(f"- Recall diff   : {fe_rf_results['recall'] - fe_log_results['recall']}")
    print(f"- F1 diff       : {fe_rf_results['f1'] - fe_log_results['f1']}")
    print(f"- ROC-AUC diff  : {fe_rf_results['roc_auc'] - fe_log_results['roc_auc']}")


    print("\nTop 10 Random Forest features:")

    rf_pipeline = fe_rf_results["pipeline"]
    rf_model = rf_pipeline.named_steps["model"]
    rf_feature_names = fe_rf_results["feature_names"]

    feature_importance = pd.Series(
        rf_model.feature_importances_,
        index=rf_feature_names
    ).sort_values(ascending=False)

    print(feature_importance.head(10))


if __name__ == "__main__":
    main()