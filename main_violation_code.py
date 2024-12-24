import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

model1_name = "nn_model2"
# model2_name = "nn_model_lrscheduler_earlyStop1"

# Load data
df_og = pd.read_csv("./Data/Parking_Violations_Issued_3.csv")  # Replace with your CSV path
print(df_og.shape)
df = df_og[~df_og['Violation Code'].isin([0, 95])]
df = df[df['Plate Type'] != '999']

print(df.columns)
fill_cols = ['Vehicle Make', 'Vehicle Body Type', 'Vehicle Color', 'Street Code2', 'Street Code2',
             'Violation County', 'Street Code1', 'Plate Type', 'Issuing Agency',
             'Violation Precinct', 'Registration State', 'Violation Time']

for col in fill_cols:
    print("{}: {}".format(col, len(df[col].unique())))
print(len(df["Street Code1"].unique()))
print(len(df["Street Code2"].unique()))
print(len(df["Street Code3"].unique()))

# CODE TO MAP FINE AMOUNTS BASED ON VIOLATION CODE AND LOCATION:
# Map Fine Amounts to Violation Code, since it will be target variable:
county_mapping = {
    'K': 'Brooklyn',
    'Q': 'Queens',
    'BX': 'Bronx',
    'R': 'Staten Island',
    'NY': 'Manhattan',
    'MN': 'Manhattan',
    'Bronx': 'Bronx',
    'Qns': 'Queens',
    'Kings': 'Brooklyn',
    'Rich': 'Staten Island',
    'BK': 'Brooklyn',
    'QN': 'Queens',
    'ST': 'Staten Island',
    'QNS': 'Queens',
    '108': 'Unknown',
    None: 'Unknown'
}

# Apply mapping
df['Violation County Normalized'] = df['Violation County'].map(county_mapping)

# Load fine mapping dataset
fine_mapping = pd.read_csv('./Data/DOF_Parking_Violation_Codes.csv')

fine_mapping.loc[len(fine_mapping)] = {
    'CODE': 94,
    fine_mapping.columns[2]: 100,
    'All Other Areas': 100
}

fine_mapping.columns = fine_mapping.columns.str.strip()

# Rename columns for clarity
fine_mapping.rename(columns={
    "CODE": "Violation Code",
    fine_mapping.columns[2]: "Manhattan Fine",
    "All Other Areas": "Other Areas Fine"
}, inplace=True)

# Merge fine amounts into main dataframe
df = df.merge(fine_mapping, on='Violation Code', how='left')

# Check for rows where fine amounts are missing
missing_fine_rows = df[df['Manhattan Fine'].isnull() & df['Other Areas Fine'].isnull()]
print(f"Number of rows with missing fine amounts: {len(missing_fine_rows)}")

# Map fine amount based on normalized county
df['Fine Amount'] = df.apply(
    lambda row: row['Manhattan Fine'] if row['Violation County Normalized'] == 'Manhattan' else row['Other Areas Fine'],
    axis=1
)

print(len(df['Vehicle Color'].unique()))

# Normalize to uppercase
df['Vehicle Color'] = df['Vehicle Color'].str.upper()

# Define an expanded mapping dictionary
color_mapping = {
    'BK': 'BLACK', 'BLK': 'BLACK', 'BLA': 'BLACK', 'BLAK': 'BLACK',
    'BL': 'BLUE', 'DKB': 'BLUE', 'BLU': 'BLUE', 'LTB': 'BLUE', 'BLE': 'BLUE',
    'BN': 'BROWN', 'BRN': 'BROWN', 'BR': 'BROWN', 'BRO': 'BROWN',
    'GR': 'GREEN', 'GN': 'GREEN', 'GRN': 'GREEN', 'GREE': 'GREEN',
    'WH': 'WHITE', 'WHT': 'WHITE', 'WHI': 'WHITE', 'WT': 'WHITE', 'WHE': 'WHITE',
    'RD': 'RED', 'MR': 'RED', 'RED': 'RED', 'DKR': 'RED',
    'SL': 'SILVER', 'SILVR': 'SILVER', 'SILV': 'SILVER', 'SILVER': 'SILVER',
    'GY': 'GRAY', 'GRY': 'GRAY', 'GREY': 'GRAY', 'GRAY': 'GRAY', 'GEY': 'GRAY',
    'OR': 'ORANGE', 'ORANG': 'ORANGE', 'ORANGE': 'ORANGE',
    'PR': 'PURPLE', 'PURPL': 'PURPLE', 'PURP': 'PURPLE',
    'GL': 'GOLD', 'GOLD': 'GOLD',
    'YW': 'YELLOW', 'YELLO': 'YELLOW', 'YELLOW': 'YELLOW', 'YLW': 'YELLOW',
    'PK': 'PINK', 'PIK': 'PINK', 'PINK': 'PINK',
    'DK/': 'OTHER', 'LT/': 'OTHER', 'NOCL': 'OTHER', 'OTHER': 'OTHER',
    'MULTI': 'OTHER', 'NAN': 'OTHER'
}

# Replace using the mapping dictionary
df['Vehicle Color'] = df['Vehicle Color'].replace(color_mapping)

# Assign remaining uncommon values to "OTHER"
common_colors = ['BLACK', 'BLUE', 'BROWN', 'GREEN', 'WHITE', 'RED', 'SILVER', 'GRAY', 'ORANGE', 'PURPLE', 'GOLD', 'YELLOW', 'PINK']
df['Vehicle Color'] = df['Vehicle Color'].apply(lambda x: x if x in common_colors else 'OTHER')
print(len(df['Vehicle Color'].unique()))

# Fill missing values with mode

# for col in fill_cols:
#     df[col] = df[col].fillna(df[col].mode()[0])

# Try dropping NaN value rows instead of filling with mode
df = df.dropna(subset=fill_cols)
print(df.shape)
# Extract day of the week from Issue_Date
df['Issue Date'] = pd.to_datetime(df['Issue Date'])
df['Day Of Week'] = df['Issue Date'].dt.dayofweek  # 0=Monday, 6=Sunday
df.drop(columns=['Issue Date'], inplace=True)


# Helper function to convert time strings into 24-hour format
def parse_time(time_str):
    # Extract hour, minute, and AM/PM
    hour = int(time_str[:2])
    minute = int(time_str[2:4])
    period = time_str[4]  # 'A' or 'P'

    # Convert to 24-hour time
    if period == 'P' and hour != 12:
        hour += 12
    if period == 'A' and hour == 12:
        hour = 0
    return hour, minute


# Apply the parsing function
df['Hour'], df['Minute'] = zip(*df['Violation Time'].map(parse_time))

# Add derived features
df['Total Minutes'] = df['Hour'] * 60 + df['Minute']  # Minutes since midnight

# Optional: Add cyclic features
df['Hour Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)  # Hour as a cyclic feature
df['Hour Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)

unique_fine_amounts = sorted(df['Fine Amount'].unique())
fine_amount_to_index = {fine: i for i, fine in enumerate(unique_fine_amounts)}
index_to_fine_amount = {i: fine for fine, i in fine_amount_to_index.items()}  # Reverse mapping from above

df['Fine Amount Index'] = df['Fine Amount'].map(fine_amount_to_index)

# Verify the new column:
print("Unique Fine Amount Indices:", df['Fine Amount Index'].unique())
print("Fine Amount to Index Mapping:", fine_amount_to_index)

target = 'Violation Code'
features = ['Vehicle Body Type', 'Vehicle Color', 'Vehicle Make',
            'Violation County', 'Street Code1', 'Street Code2', 'Street Code3',
            'Day Of Week', 'Issuing Agency',
            'Plate Type', 'Violation Precinct', 'Registration State', 'Hour Sin', 'Hour Cos']

X = df[features]
y = df[target]
print(X.shape)

categorical_cols = ['Vehicle Body Type', 'Vehicle Color', 'Vehicle Make',
                    'Violation County', 'Street Code1', 'Issuing Agency', 'Street Code2', 'Street Code3',
                    'Plate Type', 'Violation Precinct', 'Registration State']
numerical_cols = ['Day Of Week', 'Hour Sin', 'Hour Cos']

categorical_transformer = Pipeline(steps=[
    ('label', LabelEncoder())
])

numerical_transformer = Pipeline(steps=[
    ('scale', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_cols),  # One-hot encode categorical features
        ('num', numerical_transformer, numerical_cols)
    ]
)

X = preprocessor.fit_transform(X)

print(X.shape)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Convert y to categorical (for multi-class classification)
y_train_categorical = to_categorical(y_train)
y_test_categorical = to_categorical(y_test)

# Neural Network Model Setup:
nn_model = Sequential([
    Dense(128, activation='relu', input_dim=X_train.shape[1]),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y_train_categorical.shape[1], activation='softmax')  # Output layer for multi-class classification
])

# Compile the Neural Network:
nn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Neural Network
history = nn_model.fit(X_train, y_train_categorical, epochs=20, batch_size=128, validation_data=(X_test, y_test_categorical))
# Evaluate
test_loss, test_accuracy = nn_model.evaluate(X_test, y_test_categorical)
print("Test Accuracy:", test_accuracy)

yhat = nn_model.predict(X_test)  # Predicted probabilities

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_pred_classes = np.argmax(yhat, axis=1)
y_test_classes = np.argmax(y_test_categorical, axis=1)

# Generate confusion matrix
cm = confusion_matrix(y_test_classes, y_pred_classes)

# Class labels
class_labels = [f"Class {i}" for i in range(len(cm))]  # Replace with actual class names if available

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

predicted_class_indices = np.argmax(yhat, axis=1)

# Convert predicted indices to Fine Amounts:
# predicted_fine_amounts = [index_to_fine_amount[idx] for idx in predicted_class_indices]
violation_code_to_fine_amount = pd.Series(
     fine_mapping['Manhattan Fine'].values, index=fine_mapping['Violation Code']).to_dict()

# Convert predicted violation codes to fine amounts
predicted_fine_amounts = [violation_code_to_fine_amount.get(violation_code, 0) for violation_code in predicted_class_indices]

# Convert actual indices back to Fine Amounts
actual_class_indices = np.argmax(y_test_categorical, axis=1)
# actual_fine_amounts = [violation_code_to_fine_amount.get(violation_code, 0)  # Default to 0 if violation code not found
#                        for violation_code in actual_class_indices]
actual_fine_amounts = [index_to_fine_amount[idx] for idx in actual_class_indices]

# Step 6: Compare predictions with actual values
print("Predicted Fine Amounts Revenue:", np.sum(predicted_fine_amounts))
print("Actual Fine Amounts Revenue:", np.sum(actual_fine_amounts))

# Calculate accuracy
print("Accuracy:", accuracy_score(y_pred_classes, y_test_classes))
nn_model.save(f'saved_model/{model1_name}')


def plot_train_test_hist(history, acc_plot_name, loss_plot_name):
    import matplotlib.pyplot as plt

    # Extract accuracy and loss from the history object
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Plot accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(train_accuracy, label='Train Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Over Epochs')
    plt.legend()
    plt.grid()
    # Save the accuracy plot
    plt.savefig(acc_plot_name)
    plt.show()

    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Model Loss Over Epochs')
    plt.legend()
    plt.grid()
    # Save the loss plot
    plt.savefig(loss_plot_name)
    plt.show()


plot_train_test_hist(history, "model2_acc.png", "model2_loss.png")
# plot_train_test_hist(history_2, "model2_acc.png", "model2_loss.png")
