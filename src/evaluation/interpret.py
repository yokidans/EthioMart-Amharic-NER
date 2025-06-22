# src/evaluation/interpret.py
import shap
import lime
from transformers import pipeline


def explain_model():
    nlp = pipeline("ner", model="models/fine_tuned/ethiomart_ner")
    
    # SHAP explanation
    explainer = shap.Explainer(nlp)
    shap_values = explainer(["ስልክ 5000 ብር �ዲስ አበባ"])
    
    # LIME explanation
    lime_exp = lime.lime_text.LimeTextExplainer()
    exp = lime_exp.explain_instance(
        "ስልክ 5000 ብር",
        nlp.predict_proba,
        num_features=5
    )
    
    return shap_values, exp


if __name__ == "__main__":
    explain_model()
