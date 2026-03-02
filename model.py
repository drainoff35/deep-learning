import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# 1. Verinin Yüklenmesi ve Temizlenmesi
df = pd.read_csv('data.csv')
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)
if 'Unnamed: 32' in df.columns:
    df.drop('Unnamed: 32', axis=1, inplace=True)

# 2. Hedef Değişkenin Kodlanması (Kötü huylu = 1, İyi huylu = 0)
le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])

# 3. Verinin Eğitim ve Test Olarak Ayrılması
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. YENİ YÖNTEM: Aykırı Değerlere Karşı Dirençli Ölçeklendirme
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Modelin Kurulması ve Eğitilmesi
model = LogisticRegression(C=0.1, solver='liblinear', random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Tahmin ve Metriklerin Hesaplanması
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"En İyi Doğruluk: {accuracy:.4f}")
print(f"Kesinlik: {precision:.4f}")
print(f"Duyarlılık: {recall:.4f}")
print(f"F1-Skoru: {f1:.4f}")
print("\nHata Matrisi:")
print(conf_matrix)
