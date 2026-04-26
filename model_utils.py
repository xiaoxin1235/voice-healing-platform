import numpy as np

def get_deep_features(audio, sr=16000):
    return np.random.rand(768)

def assess_depression_risk(acoustic_features, deep_features):
    score = 0
    if acoustic_features["zero_cross_rate"] < 0.05:
        score += 20
    if acoustic_features["f0_mean"] < 120:
        score += 25
    if acoustic_features["rms_energy"] < 0.02:
        score += 25
    if acoustic_features["mel_spec_mean"] < 0.01:
        score += 30

    risk_score = min(score, 100)
    if risk_score < 30:
        level = "低风险"
        suggestion = "心理状态平稳，保持健康作息即可。"
    elif 30 <= risk_score < 60:
        level = "中风险"
        suggestion = "情绪存在轻微低落，建议多放松、增加社交。"
    else:
        level = "高风险"
        suggestion = "情绪压力较高，建议及时疏导或咨询专业心理人员。"

    return {
        "risk_score": risk_score,
        "risk_level": level,
        "suggestion": suggestion
    }