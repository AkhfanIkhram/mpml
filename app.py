import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Mengatur tema dan konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Status Transaksi UPI",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Fungsi untuk menambahkan CSS kustom
def local_css(file_name):
    """Loads local CSS file for styling."""
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"File {file_name} not found. Please check the file path.")

# Panggil fungsi CSS dengan path yang benar
local_css("style.css")  # Make sure style.css is in the same directory or provide correct path

# Memuat model yang disimpan
try:
    best_model = joblib.load('best_model.pkl')
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Memuat dataset untuk mendapatkan informasi prapemrosesan
df = pd.read_csv('transactions.csv')

# Menyederhanakan fitur numerik
scaler = StandardScaler()
df[['Amount (INR)']] = scaler.fit_transform(df[['Amount (INR)']])

# Mengkodekan UPI ID (karena mereka bisa berupa string)
label_encoder_sender = LabelEncoder()
label_encoder_sender.fit(df['Sender UPI ID'])

label_encoder_receiver = LabelEncoder()
label_encoder_receiver.fit(df['Receiver UPI ID'])

# Fungsi untuk memperluas LabelEncoder dengan label baru
def extend_label_encoder(label_encoder, new_label):
    """Extend LabelEncoder with new labels."""
    if new_label not in label_encoder.classes_:
        # Perluas label encoder dengan label baru
        new_classes = np.append(label_encoder.classes_, new_label)
        label_encoder.classes_ = new_classes

# Fungsi prediksi menggunakan model yang disimpan
def predict(sender_upi_id, receiver_upi_id, amount_inr):
    """Predict transaction status using the loaded model."""
    # Periksa dan tambahkan label baru jika perlu
    extend_label_encoder(label_encoder_sender, sender_upi_id)
    extend_label_encoder(label_encoder_receiver, receiver_upi_id)
    
    # Menskalakan input
    scaled_amount = scaler.transform([[amount_inr]])
    # Mengkodekan input
    encoded_sender = label_encoder_sender.transform([sender_upi_id])
    encoded_receiver = label_encoder_receiver.transform([receiver_upi_id])
    # Membuat prediksi
    features = np.array([[encoded_sender[0], encoded_receiver[0], scaled_amount[0][0]]])
    prediction = best_model.predict(features)
    return prediction[0]

# Membuat antarmuka Streamlit
st.title("Prediksi Status Transaksi UPI")

# Membuat satu baris dengan tiga kolom
col1, col2, col3 = st.columns([2, 2, 1])  # Menyesuaikan lebar kolom

with col1:
    sender_upi_id = st.text_input("Sender UPI ID:", key="sender_upi_id")

with col2:
    receiver_upi_id = st.text_input("Receiver UPI ID:", key="receiver_upi_id")

with col3:
    amount_inr = st.number_input("Amount (INR):", min_value=0.0, step=0.01, key="amount_inr")

st.markdown("---")  # Garis pemisah

if st.button("Prediksi"):
    sender_upi_id = st.session_state.sender_upi_id
    receiver_upi_id = st.session_state.receiver_upi_id
    amount_inr = st.session_state.amount_inr

    try:
        result = predict(sender_upi_id, receiver_upi_id, amount_inr)
        if result == 1:
            st.balloons()
            st.success("Transaksi Sukses")
        else:
            st.snow()
            st.error("Transaksi Gagal")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
