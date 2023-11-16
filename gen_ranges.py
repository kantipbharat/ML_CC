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

csv_name = 'datasets/' + version_name + '.csv'
df = pd.read_csv(csv_name, index_col=0); df = df.dropna()

rl_status_cols = ['ewma_inter_send', 'ewma_inter_arr', 'ratio_rtt', 'ssthresh', 'cwnd']

ranges = {}
data = df[rl_status_cols]
for column in rl_status_cols:
    first = np.percentile(data[column], 1)
    last = np.percentile(data[column], 99)
    ranges[column] = (first, last)

ranges_name = 'objects/' + version_name + '.pkl'
if os.path.exists(ranges_name): os.remove(ranges_name)
pickle.dump(ranges, open(ranges_name, 'wb'))