import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_classif, chi2, mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#====== Univariate feature
# SelectKBest
selector = SelectKBest(score_func=f_classif, k=5) # f_classif for classification, f_regression for regression
X_new = selector.fit_transform(X, y)

# SelectPercentile
selector = SelectPercentile(score_func=mutual_info_classif, percentile=50) # mutual_info_classif for classification, mutual_info_regression for regression
X_new = selector.fit_transform(X, y)



#=======Recursive feature elimination (RFE)
estimator = LogisticRegression(solver='liblinear') # You can use other models as well
selector = RFE(estimator, n_features_to_select=5, step=1)
X_new = selector.fit_transform(X, y)


#=======Feature importance based on tree-based models:
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)
importances = model.feature_importances_
important_features = np.argsort(importances)[::-1][:5] # Top 5 features
X_new = X[:, important_features]


#=======Lasso regulation
model = Lasso(alpha=0.1)
model.fit(X, y)
important_features = np.nonzero(model.coef_)[0]
X_new = X[:, important_features]


#=============Ridge regularization:
model = Ridge(alpha=1)
model.fit(X, y)
coef_abs = np.abs(model.coef_)
important_features = np.argsort(coef_abs)[::-1][:5] # Top 5 features
X_new = X[:, important_features]



#=============Elastic Net regularization:
model = ElasticNet(alpha=1, l1_ratio=0.5)
model.fit(X, y)
important_features = np.nonzero(model.coef_)[0]
X_new = X[:, important_features]




#============Correlation-based feature selection:
correlations = X.corrwith(y)
important_features = correlations.abs().nlargest(5).index
X_new = X.loc[:, important_features]




#===========Variance-based feature selection:
threshold = 0.1
low_var_features = X.var() < threshold
X_new = X.loc[:, ~low_var_features]




#============Principal Component Analysis (PCA):
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=5) # Choose the number of principal components to keep
X_new = pca.fit_transform(X_scaled)