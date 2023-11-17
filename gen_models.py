from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from helper import *

if len(sys.argv) == 1:
    print("Must include version!"); exit(1)

version = sys.argv[1]; version_name = ''
if version in VERSION_MAP.keys(): version_name = VERSION_MAP[version]
else:
    print("Invaid version!"); exit(1)

df_name = 'data/dataframes/' + version_name + '.pkl'
df = pickle.load(open(df_name, 'rb')); df = df.dropna()

cols_to_remove = ['num', 'idx', 'send_time'] + ['ssthresh', 'throughput', 'max_throughput', 'loss_rate', 'overall_loss_rate', 'delay']
cols_to_remove += ['ratio_inter_send', 'ratio_inter_arr', 'ratio_rtt']

df = df.drop(cols_to_remove, axis=1)
X = df.iloc[:, :-1]; y = df.iloc[:, -1]

# drops random data to perform grid search on a small portion of data to find best parameters
# X, X_drop, y, y_drop = train_test_split(X, y, test_size=0.999, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

# parameter grid for testing with small sets of data
param_grid = {
    'bootstrap': [True, False],
    'max_depth': [10, 20, 50, 100, None],
    'min_samples_leaf': [1, 2, 5],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [10, 20, 50, 100]
 }

# found that most parameters don't make a significant difference, only bootstrap=True improved performance significantly
rf_model = RandomForestClassifier(random_state=42, bootstrap=True, n_estimators=50, max_depth=100, n_jobs=-1)

rf_model.fit(X_train, y_train)
best_model = rf_model

# grid_search = GridSearchCV(rf_model, param_grid, cv=10, scoring='accuracy', verbose=2)
# grid_search.fit(X_train, y_train)
# best_model = grid_search.best_estimator_
# print("Best Parameters: " + str(grid_search.best_params_))
# print("Best Score: " + str(grid_search.best_score_))

# cv_results = grid_search.cv_results_
# for mean_score, params in zip(cv_results["mean_test_score"], cv_results["params"]):
#     print(f"Mean accuracy: {mean_score:.3f} for Parameters: {params}")

y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
train_accuracy = accuracy_score(y_train, best_model.predict(X_train))

print("Best Test Accuracy: " + str(test_accuracy * 100) + "%")
print("Best Train Accuracy: " + str(train_accuracy * 100) + "%")

model_name = 'data/models/' + version_name + '.pkl'
if os.path.exists(model_name): os.remove(model_name)
pickle.dump(best_model, open(model_name, 'wb'))