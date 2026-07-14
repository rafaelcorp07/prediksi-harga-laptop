import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# 1. Konfigurasi Halaman Web
st.set_page_config(page_title="Estimasi Harga Laptop", layout="wide")

# 2. Fungsi Load Model & Data Bersih (Menggunakan Cache agar Web Cepat)
@st.cache_resource
def load_ml_components():
    with open('model_laptop_rf.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_clean_data():
    return pd.read_csv('laptop_price_clean.csv')

# Panggil Komponen ML & Data
komponen = load_ml_components()
model = komponen['model']
scaler = komponen['scaler']
fitur_kolom = komponen['fitur_kolom']

df = load_clean_data()

# 3. Antarmuka Utama (Header)
st.title("Dashboard Analisis & Prediksi Harga Laptop")
st.markdown("Aplikasi ini menggunakan kecerdasan **Random Forest Regressor** untuk memprediksi harga laptop berdasarkan spesifikasi yang Anda pilih.")
st.write("---")

# 4. Form Input Spesifikasi di Sidebar (Panel Kiri)
st.sidebar.header("Spesifikasi Laptop")

company = st.sidebar.selectbox("Merek Laptop", sorted(df['Company'].unique()))
typename = st.sidebar.selectbox("Jenis/Kategori", sorted(df['TypeName'].unique()))
inches = st.sidebar.slider("Ukuran Layar (Inches)", float(df['Inches'].min()), float(df['Inches'].max()), 15.6)
ram = st.sidebar.selectbox("Kapasitas RAM (GB)", sorted(df['Ram'].unique()), index=2) # Default 8GB
weight = st.sidebar.slider("Berat Laptop (kg)", float(df['Weight'].min()), float(df['Weight'].max()), 2.0)
cpu_brand = st.sidebar.selectbox("Brand CPU", sorted(df['Cpu_Brand'].unique()))
cpu_speed = st.sidebar.slider("Kecepatan CPU (GHz)", float(df['Cpu_Speed_GHz'].min()), float(df['Cpu_Speed_GHz'].max()), 2.5)
ssd = st.sidebar.selectbox("Kapasitas SSD (GB)", sorted(df['SSD_GB'].unique()))
hdd = st.sidebar.selectbox("Kapasitas HDD (GB)", sorted(df['HDD_GB'].unique()))
touchscreen = st.sidebar.radio("Layar Sentuh?", ["Tidak", "Ya"])
ips = st.sidebar.radio("Panel IPS?", ["Tidak", "Ya"])
opsys = st.sidebar.selectbox("Sistem Operasi", sorted(df['OpSys'].unique()))
gpu_brand = st.sidebar.selectbox("Brand GPU", sorted(df['Gpu_Brand'].unique()))

# Resolusi bawaan standar pasar
res_width = 1920
res_height = 1080

# 5. Logika Prediksi Saat Tombol Ditekan
if st.sidebar.button("Hitung Estimasi Harga"):
    # Tampung input ke dalam dictionary
    input_user = {
        'Inches': inches, 'Ram': ram, 'Weight': weight, 'Cpu_Speed_GHz': cpu_speed,
        'SSD_GB': ssd, 'HDD_GB': hdd, 
        'Touchscreen': 1 if touchscreen == "Ya" else 0,
        'IPS_Panel': 1 if ips == "Ya" else 0,
        'Resolution_Width': res_width, 'Resolution_Height': res_height,
        'Company': company, 'TypeName': typename, 'Cpu_Brand': cpu_brand,
        'Gpu_Brand': gpu_brand, 'OpSys': opsys
    }
    
    # Ubah ke DataFrame & lakukan Encoding
    df_input = pd.DataFrame([input_user])
    df_input_encoded = pd.get_dummies(df_input)
    
    # Sinkronisasi kolom agar persis sama dengan saat training di Colab
    df_final = pd.DataFrame(columns=fitur_kolom)
    for col in fitur_kolom:
        if col in df_input_encoded.columns:
            df_final.at[0, col] = df_input_encoded.at[0, col]
        else:
            df_final.at[0, col] = 0
            
    df_final = df_final.astype(float)
    
    # Lakukan Penskalaan (Scaling) data input
    df_final_scaled = scaler.transform(df_final)
    
    # Prediksi
    prediksi_euro = model.predict(df_final_scaled)[0]
    prediksi_rupiah = prediksi_euro * 17500 # Estimasi kurs Euro ke Rupiah
    
    # Tampilkan Hasil Prediksi Utama
    st.subheader("Hasil Estimasi Harga")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Harga dalam Euro", value=f"€ {prediksi_euro:,.2f}")
    with col2:
        st.metric(label="Harga Perkiraan (Rupiah)", value=f"Rp {prediksi_rupiah:,.0f}")
    st.success("Prediksi berhasil dihitung!")

st.write("---")

# 6. Fitur Tambahan: Visualisasi Singkat di Layar Utama
st.subheader("Analisis Singkat Pasar Laptop")
pilihan_grafik = st.selectbox("Pilih Grafik Visualisasi:", ["Distribusi Harga per Merek", "Kapasitas RAM vs Harga"])

if pilihan_grafik == "Distribusi Harga per Merek":
    fig = px.box(df, x="Company", y="Price_in_euros", title="Rentang Harga Laptop Berdasarkan Merek", color="Company")
    st.plotly_chart(fig, use_container_width=True)
elif pilihan_grafik == "Kapasitas RAM vs Harga":
    df_ram = df.groupby("Ram")["Price_in_euros"].mean().reset_index()
    fig = px.bar(df_ram, x="Ram", y="Price_in_euros", title="Rata-rata Harga Berdasarkan Kapasitas RAM (GB)", labels={"Price_in_euros": "Rata-rata Harga (Euro)"})
    st.plotly_chart(fig, use_container_width=True)
