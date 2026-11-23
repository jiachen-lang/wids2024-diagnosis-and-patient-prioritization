"""Competition analysis: explore, compare two settings, evaluate, submit."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.calibration import calibration_curve

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports/tables"
IMAGES = ROOT / "reports/images"
TARGET = "DiagPeriodL90D"
SEED = 42
EXCLUDED_COLUMNS = [
    TARGET, "patient_id", "patient_gender",
    "metastatic_first_novel_treatment", "metastatic_first_novel_treatment_type",
]
CANDIDATES = [
    dict(iterations=150, depth=3, learning_rate=0.08, l2_leaf_reg=5),
    dict(iterations=250, depth=4, learning_rate=0.06, l2_leaf_reg=7),
]


def read_data():
    train = pd.read_csv(ROOT / "input/training.csv")
    test = pd.read_csv(ROOT / "input/test.csv")
    sample = pd.read_csv(ROOT / "input/sample_submission.csv")
    assert set(train.columns) - {TARGET} == set(test.columns)
    assert train[TARGET].isin([0, 1]).all()
    assert train.patient_id.is_unique and test.patient_id.is_unique
    assert set(train.patient_id).isdisjoint(test.patient_id)
    assert sample.patient_id.equals(test.patient_id)
    TABLES.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    return train, test, sample


def explore_data(train, test):
    """Describe data quality and observed insurance differences."""
    missing = pd.DataFrame({
        "training_missing_pct": train.drop(columns=TARGET).isna().mean() * 100,
        "test_missing_pct": test.isna().mean() * 100,
    })
    missing.sort_values("training_missing_pct", ascending=False).to_csv(TABLES / "missingness.csv")
    groups = train.assign(insurance=train.payer_type.fillna("Unknown"))
    groups = groups.groupby("insurance")[TARGET].agg(["count", "mean"])
    groups.columns = ["patients", "timely_diagnosis_rate"]
    groups.to_csv(TABLES / "insurance_descriptive.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    missing.nlargest(8, "training_missing_pct").iloc[::-1].plot.barh(ax=ax)
    ax.set(xlabel="Missing records (%)", ylabel="", title="Missingness in supplied data")
    fig.tight_layout()
    fig.savefig(IMAGES / "missingness.png", dpi=160)
    plt.close(fig)
    summary = dict(training_rows=len(train), test_rows=len(test), training_columns=train.shape[1],
                   timely_rate=float(train[TARGET].mean()), delayed_count=int((train[TARGET] == 0).sum()))
    (TABLES / "data_summary.json").write_text(json.dumps(summary, indent=2))
    return missing, groups


def prepare_features(frame, columns):
    """Keep numerical NaN; represent missing categories explicitly."""
    x = frame[columns].copy()
    cats = x.select_dtypes(include=["object", "string"]).columns.tolist()
    if "patient_zip3" in columns:
        cats.append("patient_zip3")
    for column in cats:
        x[column] = x[column].fillna("Unknown").astype(str)
    for column in ["bmi", "patient_race", "payer_type"]:
        x[column + "_missing"] = frame[column].isna().astype(int)
    return x, cats


def new_model(params, cats):
    return CatBoostClassifier(
        **params, cat_features=cats, loss_function="Logloss", eval_metric="AUC",
        random_seed=SEED, verbose=False, thread_count=4,
        max_ctr_complexity=1, allow_writing_files=False,
    )


def split_data(train):
    development, holdout = train_test_split(
        np.arange(len(train)), test_size=0.2, stratify=train[TARGET], random_state=SEED,
    )
    np.savez(ROOT / "reports/split_indices.npz", development=development, holdout=holdout)
    return development, holdout


def compare_candidates(train, x, cats, development):
    """Choose by development OOF AUC, without inspecting holdout outcomes."""
    y = train[TARGET]
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    fold_scores, oof_predictions, candidate_scores = [], [], []
    for number, params in enumerate(CANDIDATES):
        oof = np.zeros(len(development))
        for fold, (training, validation) in enumerate(cv.split(x.iloc[development], y.iloc[development]), 1):
            model = new_model(params, cats)
            model.fit(x.iloc[development[training]], y.iloc[development[training]])
            probability = model.predict_proba(x.iloc[development[validation]])[:, 1]
            oof[validation] = probability
            auc = roc_auc_score(y.iloc[development[validation]], probability)
            fold_scores.append(dict(model="competition", candidate=number, fold=fold, auc=auc, **params))
            print(f"Candidate {number}, fold {fold}: AUC {auc:.5f}", flush=True)
        oof_predictions.append(oof)
        candidate_scores.append(roc_auc_score(y.iloc[development], oof))
    best = int(np.argmax(candidate_scores))
    params, oof = CANDIDATES[best], oof_predictions[best]
    pd.DataFrame(fold_scores).to_csv(ROOT / "scores.csv", index=False)
    pd.DataFrame({"patient_id": train.iloc[development].patient_id.to_numpy(),
                  "actual_timely": y.iloc[development].to_numpy(), "competition_probability": oof}).to_csv(
        ROOT / "reports/development_predictions.csv", index=False,
    )
    return params, oof


def bootstrap_auc(y, probability):
    rng = np.random.default_rng(2024)
    values = []
    for _ in range(1000):
        sample = rng.integers(0, len(y), len(y))
        if len(np.unique(y[sample])) == 2:
            values.append(roc_auc_score(y[sample], probability[sample]))
    return np.quantile(values, [0.025, 0.975]).tolist()


def evaluate_holdout(train, x, cats, development, holdout, params, oof):
    """Evaluate the selected model once; save probabilities for the later audit."""
    y = train[TARGET]
    model = new_model(params, cats)
    model.fit(x.iloc[development], y.iloc[development])
    probability = model.predict_proba(x.iloc[holdout])[:, 1]
    summary = dict(
        development_oof_auc=roc_auc_score(y.iloc[development], oof),
        holdout_auc=roc_auc_score(y.iloc[holdout], probability),
        holdout_auc_ci=bootstrap_auc(y.iloc[holdout].to_numpy(), probability),
        holdout_brier=brier_score_loss(y.iloc[holdout], probability), features=x.shape[1],
    )
    (TABLES / "model_summary.json").write_text(json.dumps({"competition": summary}, indent=2))
    pd.DataFrame({
        "patient_id": train.iloc[holdout].patient_id.to_numpy(),
        "actual_timely": y.iloc[holdout].to_numpy(),
        "insurance": x.iloc[holdout].payer_type.to_numpy(),
        "competition_probability": probability,
    }).to_csv(ROOT / "reports/holdout_predictions.csv", index=False)
    importance = pd.DataFrame({"feature": x.columns, "importance": model.feature_importances_})
    importance.sort_values("importance", ascending=False).to_csv(TABLES / "competition_importance.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fpr, tpr, _ = roc_curve(y.iloc[holdout], probability)
    axes[0].plot(fpr, tpr, label=f"AUC={summary['holdout_auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].legend()
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="Competition holdout ROC")
    actual, predicted = calibration_curve(y.iloc[holdout], probability, n_bins=8, strategy="quantile")
    axes[1].plot(predicted, actual, "o-")
    axes[1].plot([0, 1], [0, 1], "--", color="gray")
    axes[1].set(xlabel="Mean predicted timely probability", ylabel="Observed timely proportion", title="Calibration")
    fig.tight_layout()
    fig.savefig(IMAGES / "competition_validation.png", dpi=160)
    plt.close(fig)
    return summary


def make_submission(train, test, sample, columns, params):
    """Average five fold models using all labeled patients."""
    x, cats = prepare_features(train, columns)
    test_x, _ = prepare_features(test, columns)
    y = train[TARGET]
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    probabilities = []
    for fold, (training, _) in enumerate(cv.split(x, y), 1):
        model = new_model(params, cats)
        model.fit(x.iloc[training], y.iloc[training])
        probabilities.append(model.predict_proba(test_x)[:, 1])
        print(f"Submission model {fold}/5 completed.", flush=True)
    submission = sample.copy()
    submission[TARGET] = np.mean(probabilities, axis=0)
    assert submission[TARGET].between(0, 1).all()
    assert submission.patient_id.equals(test.patient_id)
    (ROOT / "submissions").mkdir(exist_ok=True)
    submission.to_csv(ROOT / "submissions/submission_catboost.csv", index=False)
    return submission


if __name__ == "__main__":
    train, test, sample = read_data()
    explore_data(train, test)
    development, holdout = split_data(train)
    columns = [column for column in train.columns if column not in EXCLUDED_COLUMNS]
    x, cats = prepare_features(train, columns)
    y = train[TARGET]
    params, oof = compare_candidates(train, x, cats, development)
    settings = dict(columns=columns, parameters=params, development_oof_auc=roc_auc_score(y.iloc[development], oof))
    (TABLES / "model_settings.json").write_text(json.dumps({"competition": settings}, indent=2))
    print(evaluate_holdout(train, x, cats, development, holdout, params, oof), flush=True)
    make_submission(train, test, sample, columns, params)
