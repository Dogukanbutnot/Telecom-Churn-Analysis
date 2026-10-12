"""
Telekom Musteri Kaybi (Churn) - Autoencoder ile Anomali Tespiti
Derin Ogrenme Projesi - Tam Uctan Uca Sistem Kodu

Asamalar: 3.1 Problem Tanimi -> 3.2 Veri Tanitimi -> 3.3 Veri Kesfi ->
3.4 Veri On Isleme -> 3.5 Model Tasarimi -> 3.6 Model Egitimi ->
3.7 Model Degerlendirme -> 3.8 Sonuclarin Yorumlanmasi

Calistirmak icin: telco_churn_temiz.csv dosyasini ayni klasore koyun
(Proje 1'deki Asama 2 temizleme script'inin ciktisi).

Bagimliliklar: pandas, numpy, matplotlib, scikit-learn, tensorflow
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score,
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

RANDOM_STATE = 42

# ============================================================
# 3.1 PROBLEM TANIMI
# ============================================================
# Girdi: Telco musteri tablosu (demografik + hizmet + sozlesme + ucret).
# Hedef: Her musteri icin bir reconstruction error (anomali) skoru uretmek.
# Yaklasim: Autoencoder SADECE "No" (churn etmemis) musterilerle egitilir;
# "normal" davranis orüntüsünü ogrenir. "Yes" (churn) musteriler bu
# orüntüye uymadigi icin teorik olarak yuksek hata uretmesi beklenir.

# ============================================================
# 3.2 VERI SETININ TANITILMASI
# ============================================================
df = pd.read_csv("telco_churn_temiz.csv")
print("=== 3.2 Veri Seti ===")
print("Boyut:", df.shape)
print("Hedef dagilimi:\n", df["Churn"].value_counts(normalize=True).round(3))

# ============================================================
# 3.3 VERI KESFI (ozet - detayli gorseller ayri script'te)
# ============================================================
numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
print("\n=== 3.3 Veri Kesfi (ozet) ===")
print("Sayisal ozellik korelasyonu:\n", df[numeric_cols].corr().round(2))
print("Contract dagilimi (No):\n", df[df.Churn == "No"]["Contract"].value_counts(normalize=True).round(3))
print("Contract dagilimi (Yes):\n", df[df.Churn == "Yes"]["Contract"].value_counts(normalize=True).round(3))

# ============================================================
# 3.4 VERI ON ISLEME
# ============================================================
print("\n=== 3.4 Veri On Isleme ===")

# Eksik / aykiri deger kontrolu
print("Eksik hucre sayisi:", df.isnull().sum().sum())
for col in numeric_cols:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((df[col] < low) | (df[col] > high)).sum()
    print(f"  {col}: aykiri deger sayisi = {n_out}")

# Encoding
y_churn = df["Churn"]
X = df.drop(columns=["customerID", "Churn"])
X_encoded = pd.get_dummies(X, drop_first=True)
feature_columns = list(X_encoded.columns)
print("Encode sonrasi ozellik sayisi:", len(feature_columns))

# Veri bolme: Autoencoder sadece "No" grubuyla egitilir
X_no = X_encoded[y_churn == "No"].reset_index(drop=True)
X_yes = X_encoded[y_churn == "Yes"].reset_index(drop=True)

X_no_train, X_no_temp = train_test_split(X_no, test_size=0.3, random_state=RANDOM_STATE)
X_no_val, X_no_test = train_test_split(X_no_temp, test_size=0.5, random_state=RANDOM_STATE)

print("X_no_train:", X_no_train.shape, " X_no_val:", X_no_val.shape,
      " X_no_test:", X_no_test.shape, " X_yes:", X_yes.shape)

# Scaling: sadece X_no_train ile fit (veri sizintisini onlemek icin)
scaler = StandardScaler()
scaler.fit(X_no_train[numeric_cols])


def scale_numeric(frame):
    out = frame.copy()
    out[numeric_cols] = scaler.transform(out[numeric_cols])
    return out


X_no_train_s = scale_numeric(X_no_train)
X_no_val_s = scale_numeric(X_no_val)
X_no_test_s = scale_numeric(X_no_test)
X_yes_s = scale_numeric(X_yes)

# Nihai degerlendirme seti: normal-test (0) + tum Yes (1)
X_eval = pd.concat([X_no_test_s, X_yes_s], ignore_index=True)
y_eval = np.array([0] * len(X_no_test_s) + [1] * len(X_yes_s))
print("X_eval:", X_eval.shape, " y_eval dagilimi (0=normal,1=anomali):", np.bincount(y_eval))

# ============================================================
# 3.5 MODEL SECIMI / TASARIMI
# ============================================================
print("\n=== 3.5 Model Tasarimi ===")
input_dim = X_no_train_s.shape[1]

model = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(16, activation="relu", name="encoder_1"),
    layers.Dense(8, activation="relu", name="encoder_2"),
    layers.Dense(4, activation="relu", name="bottleneck"),
    layers.Dense(8, activation="relu", name="decoder_1"),
    layers.Dense(16, activation="relu", name="decoder_2"),
    layers.Dense(input_dim, activation="linear", name="output"),
])
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
model.summary()

# ============================================================
# 3.6 MODEL EGITIMI
# ============================================================
print("\n=== 3.6 Model Egitimi ===")
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

history = model.fit(
    X_no_train_s, X_no_train_s,
    validation_data=(X_no_val_s, X_no_val_s),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=2,
)

print("Durdugu epoch:", len(history.history["loss"]))
print("Son train loss:", round(history.history["loss"][-1], 4))
print("Son val loss:", round(history.history["val_loss"][-1], 4))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(history.history["loss"], label="Egitim (train) loss", color="#4C72B0")
ax.plot(history.history["val_loss"], label="Validasyon (val) loss", color="#DD5555")
ax.set_title("Autoencoder Egitim Gecmisi (Loss Egrisi)")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.legend()
plt.tight_layout()
plt.savefig("gorsel_loss_egrisi.png", dpi=130)
plt.close()

# ============================================================
# 3.7 MODEL DEGERLENDIRME
# ============================================================
print("\n=== 3.7 Model Degerlendirme ===")

val_recon = np.asarray(model.predict(X_no_val_s, verbose=0), dtype=np.float64)
val_mse = np.mean(np.square(X_no_val_s.values.astype(np.float64) - val_recon), axis=1)
threshold = np.percentile(val_mse, 95)
print(f"Esik (threshold, val setinin 95. persentili): {threshold:.4f}")

eval_recon = np.asarray(model.predict(X_eval, verbose=0), dtype=np.float64)
eval_mse = np.mean(np.square(X_eval.values.astype(np.float64) - eval_recon), axis=1)
y_pred = (eval_mse > threshold).astype(int)

acc = accuracy_score(y_eval, y_pred)
prec = precision_score(y_eval, y_pred)
rec = recall_score(y_eval, y_pred)
f1 = f1_score(y_eval, y_pred)
cm = confusion_matrix(y_eval, y_pred)
auc = roc_auc_score(y_eval, eval_mse)

print(f"Accuracy:  {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall:    {rec:.3f}")
print(f"F1-score:  {f1:.3f}")
print(f"ROC-AUC:   {auc:.3f}")
print("Confusion Matrix (satir=gercek, sutun=tahmin):\n", cm)

# Gorsel: reconstruction error dagilimi, siniflara gore
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(eval_mse[y_eval == 0], bins=40, alpha=0.6, label="Normal (No)", color="#4C72B0", density=True)
ax.hist(eval_mse[y_eval == 1], bins=40, alpha=0.6, label="Anomali (Yes/Churn)", color="#DD5555", density=True)
ax.axvline(threshold, color="black", linestyle="--", label=f"Esik = {threshold:.3f}")
ax.set_title("Reconstruction Error (MSE) Dagilimi - Siniflara Gore")
ax.set_xlabel("Reconstruction Error (MSE)")
ax.set_ylabel("Yogunluk")
ax.set_xlim(0, 0.4)
ax.legend()
plt.tight_layout()
plt.savefig("gorsel_recon_error_dagilim.png", dpi=130)
plt.close()

# Gorsel: ozellik bazinda hata (kok neden analizi icin)
per_feature_mse = pd.Series(np.mean(np.square(X_eval.values.astype(np.float64) - eval_recon), axis=0),
                             index=feature_columns)
top10 = per_feature_mse.sort_values(ascending=False).head(10)
colors = ["#4C72B0" if c in numeric_cols else "#DD5555" for c in top10.index]

fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(top10.index[::-1], top10.values[::-1], color=colors[::-1])
ax.set_title("Ozellik Bazinda Ortalama Reconstruction Error (En Yuksek 10)")
ax.set_xlabel("Ortalama Kare Hata (MSE katkisi)")
plt.tight_layout()
plt.savefig("gorsel_ozellik_bazinda_hata.png", dpi=130)
plt.close()

cat_cols = [c for c in feature_columns if c not in numeric_cols]
num_share = per_feature_mse[numeric_cols].sum() / per_feature_mse.sum() * 100
cat_share = per_feature_mse[cat_cols].sum() / per_feature_mse.sum() * 100
print(f"\nToplam hataya sayisal ozelliklerin katkisi: %{num_share:.1f}")
print(f"Toplam hataya kategorik ozelliklerin katkisi: %{cat_share:.1f}")

# ============================================================
# 3.8 SONUCLARIN YORUMLANMASI
# ============================================================
print("\n=== 3.8 Sonuclarin Yorumlanmasi ===")
print(f"""
ROC-AUC = {auc:.3f} (rastgeleden bile dusuk) -> reconstruction error, Normal (No)
ve Anomali (Yes) siniflarini ayirt edemiyor. Kok neden: kategorik (one-hot)
sutunlar toplam hatanin %{cat_share:.1f}'ini olustururken, gercek churn
sinyali tasiyan sayisal sutunlar sadece %{num_share:.1f}'ini olusturuyor.
Bu dengesizlik, dusuk-bilgili kategorik gurultunun asil sinyali bogdugunu
gosteriyor.

Sonuc: "churn = anomali" hipotezi bu haliyle dogrulanmadi. Etiket zaten
mevcut oldugu icin (Churn sutunu), denetimli bir yaklasim (orn. Proje 1'deki
DecisionTreeClassifier, test accuracy 0.794) bu problem icin daha uygun bir
secimdi. Gelistirme onerileri: ozellik agirliklandirma, composite loss
(kategorik icin binary_crossentropy + sayisal icin MSE), ve/veya sadece
guclu churn sinyali tasiyan ozelliklerle (Contract, tenure, InternetService,
MonthlyCharges) daha odakli bir model.
""")

model.save("autoencoder.keras")
print("Model kaydedildi: autoencoder.keras")
