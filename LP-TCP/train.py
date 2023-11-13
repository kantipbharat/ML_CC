from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from helper import *

start_time = time.time()

df = pd.read_csv(CSV_NAME, index_col=0); df = df.iloc[:, 2:]
X = df.iloc[:, :-1]; y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf_model, param_grid, cv=10, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
train_accuracy = accuracy_score(y_train, best_model.predict(X_train))

print("Best Parameters: " + str(grid_search.best_params_))
print("Best Score: " + str(grid_search.best_score_))
print("Best Test Accuracy: " + str(test_accuracy * 100) + "%")
print("Best Train Accuracy: " + str(train_accuracy * 100) + "%")

print(time.time() - start_time)

if os.path.exists(PKL_NAME): os.remove(PKL_NAME)
pickle.dump(best_model, open(PKL_NAME, 'wb'))
