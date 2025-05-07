import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# HERE DATASET IS BEING LOADED
def load_data(file_path):
    # Read CSV file
    df = pd.read_csv(file_path, encoding='utf-8', quotechar='"', skip_blank_lines=True)   
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')] # Remove unnamed columns     
    df.columns = [col.strip() for col in df.columns] # Clean up the headers   
    df.dropna(how='all', inplace=True)  # Drop any empty rows

    return df

# --- Load & preprocess ---
df = load_data(r"C:\Users\mawan\OneDrive - University of Zululand - Students\UNIZULU\Year 3\Semester 1\Advanced Programming Techniques\PROJECT 2\water_survey.csv")

df.columns = [
    "timestamp", "role", "residence", "shortage_frequency", "shortage_impact", "class_disruption", "trust_water_quality", "illness_experience",
    "illness_type", "access_alternative","suggestions"]

# Drop timestamp (not useful for prediction)
df.drop(columns=["timestamp"], inplace=True)

# Drop rows with missing values (optional, can be refined)
df.dropna(inplace=True)

label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

X = df.drop("illness_experience", axis=1)
y = df["illness_experience"]

# Train and test split the dataset cause yabo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TRAIN THE MIDEL HERE
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# SAVE MODEL AND IT'S ENCODERS
joblib.dump(clf, 'water_risk_model.pkl')
joblib.dump(label_encoders, 'encoders.pkl')

print("✅ Model trained and saved.")
