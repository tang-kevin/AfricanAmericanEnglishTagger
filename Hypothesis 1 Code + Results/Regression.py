import pandas as pd
import numpy as np
import statsmodels.api as sm

file_path = "file path"
data = pd.read_csv(file_path)
data['True_Label'] = data['True_Label'].replace(-1, 0)
data['Prediction'] = data['Prediction'].replace(-1, 0)

def calculate_recency_bias(index, predictions, window_size=5):
    if index == 0:
        return 0
    if index < window_size:
        return np.mean(predictions[:index] == predictions[index])
    else:
        return np.mean(predictions[index - window_size:index] == predictions[index])

data['Recency_Bias'] = [calculate_recency_bias(i, data['Prediction'].values) for i in range(len(data))]
x = data[['Recency_Bias', 'True_Label']]
y = data['Prediction']
x = sm.add_constant(x)
logit_model = sm.Logit(y, x).fit()

conf_int = logit_model.conf_int()
results = pd.DataFrame({
    'Coef.': round(logit_model.params, 6),
    'Std. Err.': round(logit_model.bse, 6),
    'Z': round(logit_model.tvalues, 6),
    'P > |Z|': round(logit_model.pvalues, 6),
    '[0.025': round(conf_int[0], 6),
    '0.975]': round(conf_int[1], 6)
})
print(results)

print("\nSummary Metrics:")
print(f"Dependent Variable: Prediction Label")
print(f"No. Observations: {len(data)}")
print(f"Dataframe Residuals: {logit_model.df_resid}")
print(f"Dataframe Model: {logit_model.df_model}")
print(f"Pseudo R²: {logit_model.prsquared}")
print(f"Log Likelihood: {logit_model.llf}")
print(f"LL-Null: {logit_model.llnull}")
print(f"LLR P-value: {logit_model.llr_pvalue}")