"""Retrospective audit of the fixed competition model by recorded insurance."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
PRIMARY_RECALL = 0.8
RECALL_TARGETS = [0.7, 0.8, 0.9]
MIN_GROUP_DELAYED = 30


def load_audit_data():
    """Read prediction artifacts from the competition step; never fit a model here."""
    required = ["reports/development_predictions.csv", "reports/holdout_predictions.csv",
                "reports/split_indices.npz", "reports/tables/model_settings.json",
                "reports/tables/model_summary.json"]
    for name in required:
        if not (ROOT / name).exists():
            raise FileNotFoundError(f"Missing {name}. Run notebook 02 or python src/analysis.py first.")
    data = pd.read_csv(ROOT / "input/training.csv")
    split = np.load(ROOT / "reports/split_indices.npz")
    development = data.iloc[split["development"]].reset_index(drop=True)
    holdout = data.iloc[split["holdout"]].reset_index(drop=True)
    for frame, name in [(development, "development"), (holdout, "holdout")]:
        predictions = pd.read_csv(ROOT / f"reports/{name}_predictions.csv")
        if "patient_id" not in predictions:
            raise ValueError("Old prediction format. Rerun notebook 02 or python src/analysis.py.")
        assert np.array_equal(predictions.patient_id, frame.patient_id), "Prediction patient order differs from the data split."
        assert np.array_equal(predictions.actual_timely, frame.DiagPeriodL90D)
        assert predictions.competition_probability.between(0, 1).all()
        frame["risk"] = 1 - predictions.competition_probability.to_numpy()
        frame["delayed"] = 1 - frame.DiagPeriodL90D
        frame["insurance"] = frame.payer_type.fillna("Unknown")
    settings = json.loads((ROOT / "reports/tables/model_settings.json").read_text())["competition"]
    summary = json.loads((ROOT / "reports/tables/model_summary.json").read_text())["competition"]
    assert np.isclose(roc_auc_score(development.delayed, development.risk), settings["development_oof_auc"], rtol=0, atol=1e-9)
    assert np.isclose(roc_auc_score(holdout.delayed, holdout.risk), summary["holdout_auc"], rtol=0, atol=1e-12)
    return development, holdout


def zip_group_counts(frame, flagged):
    """Counts used for ZIP-cluster resampling, in the same group order each time."""
    local = frame[["patient_zip3", "insurance", "delayed"]].copy()
    local["timely"] = 1 - local.delayed
    local["fn"] = ((local.delayed == 1) & ~flagged).astype(int)
    local["fp"] = ((local.delayed == 0) & flagged).astype(int)
    grouped = local.groupby(["patient_zip3", "insurance"])[["delayed", "timely", "fn", "fp"]].sum()
    levels = sorted(frame.insurance.unique())
    return {column: grouped[column].unstack(fill_value=0).reindex(columns=levels, fill_value=0).to_numpy()
            for column in ["delayed", "timely", "fn", "fp"]}


def wilson_interval(events, total):
    """A descriptive 95% interval for a binomial proportion."""
    if total == 0:
        return np.nan, np.nan
    z = 1.959963984540054
    proportion = events / total
    center = (proportion + z * z / (2 * total)) / (1 + z * z / total)
    radius = z * np.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return max(0, center - radius), min(1, center + radius)


def audit_metrics(delayed, risk, threshold):
    """Count missed delays and false alarms at a scalar or patient-level threshold."""
    delayed = np.asarray(delayed, dtype=int)
    risk = np.asarray(risk)
    flagged = risk >= threshold
    tp = int(((delayed == 1) & flagged).sum())
    fn = int(((delayed == 1) & ~flagged).sum())
    fp = int(((delayed == 0) & flagged).sum())
    tn = int(((delayed == 0) & ~flagged).sum())
    low, high = wilson_interval(fn, tp + fn)
    return dict(
        patients=len(delayed), delayed=tp + fn, timely=tn + fp,
        true_positive=tp, false_negative=fn, false_positive=fp, true_negative=tn,
        fnr=fn / (tp + fn) if tp + fn else np.nan, fnr_low=low, fnr_high=high,
        recall=tp / (tp + fn) if tp + fn else np.nan,
        fpr=fp / (tn + fp) if tn + fp else np.nan,
        precision=tp / (tp + fp) if tp + fp else np.nan,
        auc=roc_auc_score(delayed, risk) if len(np.unique(delayed)) == 2 else np.nan,
        brier=brier_score_loss(delayed, risk), mean_risk=float(risk.mean()),
        observed_delay_rate=float(delayed.mean()), small_delayed_count=tp + fn < MIN_GROUP_DELAYED,
    )


def run_insurance_audit(dev, hold):
    """Check fixed global thresholds, group errors and coding sensitivity."""
    tables = ROOT / 'reports/tables'
    targets = RECALL_TARGETS
    dev_risk = dev.risk.to_numpy()
    dev_delayed = dev.delayed.to_numpy()
    delayed = hold.delayed.to_numpy()
    hold_risk = hold.risk.to_numpy()
    insurance = hold.insurance
    reproduced_auc = roc_auc_score(dev_delayed, dev_risk)
    codes = hold.breast_cancer_diagnosis_code.fillna('').astype(str)
    coding_family = np.select([codes.str.startswith('C'), codes.str.fullmatch(r'\d+')],
                              ['C-prefixed', 'Numeric-only'], default='Other/Unknown')
    levels = sorted(insurance.unique())
    # 1. Compare the same global operating points across insurance groups.
    rows, thresholds, overall = [], [], []
    for target in targets:
        threshold=float(np.quantile(dev_risk[dev_delayed==1],1-target,method='lower'))
        achieved=float((dev_risk[dev_delayed==1]>=threshold).mean())
        thresholds.append(dict(target_development_recall=target,threshold=threshold,
            achieved_development_recall=achieved,development_auc=reproduced_auc))
        overall.append(dict(target_development_recall=target,threshold=threshold,
            **audit_metrics(delayed,hold_risk,threshold)))
        for group in levels:
            mask=insurance.eq(group).to_numpy()
            rows.append(dict(target_development_recall=target,threshold=threshold,insurance=group,
                **audit_metrics(delayed[mask],hold_risk[mask],threshold)))
    thresholds=pd.DataFrame(thresholds)
    overall=pd.DataFrame(overall)
    results=pd.DataFrame(rows)
    thresholds.to_csv(tables/'insurance_audit_thresholds.csv',index=False)
    overall.to_csv(tables/'insurance_audit_overall.csv',index=False)
    results.to_csv(tables/'insurance_audit_sensitivity.csv',index=False)
    primary=results[results.target_development_recall.eq(PRIMARY_RECALL)].copy()
    primary.to_csv(tables/'insurance_audit_groups.csv',index=False)
    threshold=float(thresholds.loc[thresholds.target_development_recall.eq(PRIMARY_RECALL),'threshold'].iloc[0])
    # 2. Check whether pooled patterns also appear within coding families.
    strata = []
    for family in sorted(np.unique(coding_family)):
        for group in levels:
            mask=(coding_family==family)&insurance.eq(group).to_numpy()
            if mask.any():
                strata.append(dict(coding_family=family,insurance=group,
                    **audit_metrics(delayed[mask],hold_risk[mask],threshold)))
    pd.DataFrame(strata).to_csv(tables/'insurance_audit_coding_strata.csv',index=False)
    # Resample ZIP-prefix clusters, keeping their patients together. Fixed model and threshold.
    counts = zip_group_counts(hold, hold_risk >= threshold)
    d, f = counts['delayed'], counts['fn']
    rng=np.random.default_rng(2026)
    replicates=[]
    for _ in range(1000):
        chosen=rng.integers(0,len(d),size=len(d))
        den=d[chosen].sum(axis=0)
        rates=np.divide(f[chosen].sum(axis=0),den,out=np.full(len(levels),np.nan),where=den>0)
        replicates.append(rates)
    replicates=np.array(replicates)
    reference=levels.index('COMMERCIAL')
    indexed=primary.set_index('insurance')
    gaps=[]
    for j,group in enumerate(levels):
        difference=replicates[:,j]-replicates[:,reference]
        valid=difference[np.isfinite(difference)]
        gaps.append(dict(insurance=group,reference='COMMERCIAL',
            fnr_difference=indexed.loc[group,'fnr']-indexed.loc['COMMERCIAL','fnr'],
            low=np.quantile(valid,.025) if len(valid) else np.nan,
            high=np.quantile(valid,.975) if len(valid) else np.nan,
            valid_replicates=len(valid),zip_clusters=len(d)))
    pd.DataFrame(gaps).to_csv(tables/'insurance_audit_fnr_gaps.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    x=np.arange(len(primary))
    axes[0].errorbar(x,primary.fnr*100,
        yerr=np.vstack([(primary.fnr-primary.fnr_low)*100,(primary.fnr_high-primary.fnr)*100]),fmt='o',capsize=4)
    axes[0].set_xticks(x,primary.insurance,rotation=15)
    axes[0].set(ylabel='Delayed patients missed (%)',ylim=(0,100),title='Primary threshold: group FNR with Wilson intervals')
    for group,part in results.groupby('insurance'):
        axes[1].plot(part.target_development_recall*100,part.fnr*100,'o-',label=group)
    axes[1].set(xlabel='Development recall target (%)',ylabel='Holdout delayed patients missed (%)',ylim=(0,100),title='Threshold sensitivity')
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT/'reports/images/insurance_audit.png',dpi=160)
    plt.close(fig)
    print(primary[['insurance','patients','delayed','false_negative','fnr','fnr_low','fnr_high','auc']].to_string(index=False),flush=True)
    return primary


def review_threshold_candidate(dev, hold):
    """Test one development-defined threshold change and record the decision."""
    tables = ROOT / 'reports/tables'
    development_risk, development_delayed = dev.risk.to_numpy(), dev.delayed.to_numpy()
    risk, delayed = hold.risk.to_numpy(), hold.delayed.to_numpy()
    development_insurance, holdout_insurance = dev.insurance.to_numpy(), hold.insurance.to_numpy()
    target = PRIMARY_RECALL
    global_threshold = float(np.quantile(development_risk[development_delayed == 1], 1 - target, method='lower'))
    # Relative research checks below were fixed before this candidate was evaluated.
    # 1. Estimate the candidate thresholds from development patients only.
    chosen = {}
    threshold_rows=[]
    for group in sorted(np.unique(development_insurance)):
        m=(development_insurance==group)&(development_delayed==1)
        threshold=float(np.quantile(development_risk[m],1-target,method='lower')) if m.sum() >= MIN_GROUP_DELAYED else global_threshold
        chosen[group]=threshold
        threshold_rows.append(dict(insurance=group,development_patients=int((development_insurance==group).sum()),
            development_delayed=int(m.sum()),threshold=threshold,
            development_recall=float((development_risk[m]>=threshold).mean()) if m.any() else np.nan,
            used_global_fallback=bool(m.sum() < MIN_GROUP_DELAYED)))
    pd.DataFrame(threshold_rows).to_csv(tables/'insurance_review_thresholds.csv',index=False)
    candidate_threshold=np.array([chosen.get(g,global_threshold) for g in holdout_insurance])
    levels=sorted(np.unique(holdout_insurance))
    group_rows=[]
    overall_rows=[]
    # 2. Compare the unchanged baseline and candidate on identical patients.
    policies={'global_baseline':np.full(len(delayed),global_threshold),'group_candidate':candidate_threshold}
    for policy,threshold in policies.items():
        overall_rows.append(dict(policy=policy,**audit_metrics(delayed,risk,threshold)))
        for group in levels:
            m=holdout_insurance==group
            group_rows.append(dict(policy=policy,insurance=group,
                threshold=float(threshold[m][0]),**audit_metrics(delayed[m],risk[m],threshold[m])))
    overall=pd.DataFrame(overall_rows)
    groups=pd.DataFrame(group_rows)
    for policy,part in groups.groupby('policy'):
        m=overall.policy.eq(policy)
        overall.loc[m,'fnr_gap']=part.fnr.max()-part.fnr.min()
        overall.loc[m,'worst_group_fnr']=part.fnr.max()
    groups.to_csv(tables/'insurance_review_groups.csv',index=False)
    overall.to_csv(tables/'insurance_review_overall.csv',index=False)
    baseline=overall.set_index('policy').loc['global_baseline']
    candidate=overall.set_index('policy').loc['group_candidate']
    checks=pd.DataFrame([
        dict(check='Smaller group FNR gap',baseline=baseline.fnr_gap,candidate=candidate.fnr_gap,
            passed=bool(candidate.fnr_gap<baseline.fnr_gap)),
        dict(check='No higher worst-group FNR',baseline=baseline.worst_group_fnr,candidate=candidate.worst_group_fnr,
            passed=bool(candidate.worst_group_fnr<=baseline.worst_group_fnr)),
        dict(check='No higher overall FPR',baseline=baseline.fpr,candidate=candidate.fpr,
            passed=bool(candidate.fpr<=baseline.fpr))])
    checks.to_csv(tables/'insurance_review_checks.csv',index=False)
    # Paired ZIP-cluster bootstrap: same sampled patients for baseline and candidate.
    clusters = [zip_group_counts(hold, risk >= threshold) for threshold in policies.values()]
    rng=np.random.default_rng(2027)
    replicates=[]
    for _ in range(1000):
        idx=rng.integers(0,len(clusters[0]['delayed']),size=len(clusters[0]['delayed']))
        measures=[]
        for cluster in clusters:
            d=cluster['delayed'][idx].sum(axis=0)
            fn=cluster['fn'][idx].sum(axis=0)
            rates=np.divide(fn,d,out=np.full(len(levels),np.nan),where=d>0)
            measures.append(np.array([np.nanmax(rates)-np.nanmin(rates),np.nanmax(rates),
                cluster['fp'][idx].sum()/cluster['timely'][idx].sum()]))
        replicates.append(measures[1]-measures[0])
    replicates=np.array(replicates)
    intervals=[]
    for j,metric in enumerate(['fnr_gap','worst_group_fnr','fpr']):
        intervals.append(dict(metric=metric,candidate_minus_baseline=candidate[metric]-baseline[metric],
            low=np.quantile(replicates[:,j],.025),high=np.quantile(replicates[:,j],.975)))
    intervals=pd.DataFrame(intervals)
    intervals.to_csv(tables/'insurance_review_changes.csv',index=False)
    # 3. Record the decision; a smaller disparity does not erase extra false alarms.
    failed_checks = checks.loc[~checks.passed, 'check'].tolist()
    decision = dict(
        research_comparison='retain_for_independent_validation' if not failed_checks else 'do_not_adopt_under_current_guardrails',
        all_relative_checks_passed=not failed_checks,
        default_policy='global_baseline',
        failed_relative_checks=failed_checks,
        reason='Failed relative research checks: ' + ', '.join(failed_checks) if failed_checks else 'Retain only for independent validation.',
        production_acceptance='not_established',
    )
    (tables/'insurance_review_decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    x=np.arange(len(levels))
    width=.35
    for j,policy in enumerate(policies):
        part=groups[groups.policy.eq(policy)].set_index('insurance').loc[levels]
        axes[0].bar(x+(j-.5)*width,part.fnr*100,width,label=policy)
        axes[1].bar(x+(j-.5)*width,part.fpr*100,width,label=policy)
    for ax,title in zip(axes,['Delayed patients missed','Timely patients falsely flagged']):
        ax.set_xticks(x,levels,rotation=15)
        ax.set(ylabel='Patients (%)',ylim=(0,100),title=title)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT/'reports/images/insurance_review.png',dpi=160)
    plt.close(fig)
    print(overall[['policy','fnr','fpr','fnr_gap','worst_group_fnr']].to_string(index=False),flush=True)
    print(checks.to_string(index=False),flush=True)
    print(json.dumps(decision,indent=2),flush=True)
    return overall,groups,checks,decision

if __name__ == '__main__':
    development, holdout = load_audit_data()
    run_insurance_audit(development, holdout)
    review_threshold_candidate(development, holdout)
