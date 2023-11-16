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

csv_name = 'data/datasets/' + version_name + '.csv'
df = pd.read_csv(csv_name, index_col=0); df = df.dropna()

df_name = 'data/dataframes/' + version_name + '.pkl'
if os.path.exists(df_name): os.remove(df_name)
pickle.dump(df, open(df_name, 'wb'))