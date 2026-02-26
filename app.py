
import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

# --- 1. Load Model and Scaler ---
@st.cache_resource
def load_model_and_scaler():
    try:
        with open('best_model_gradient_boosting.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('feature_scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        return model, scaler
    except FileNotFoundError:
        st.error("Error: Model or scaler file not found. Make sure 'best_model_gradient_boosting.pkl' and 'feature_scaler.pkl' are in the same directory.")
        st.stop()

model, scaler = load_model_and_scaler()

# --- 2. Define Global Variables (Unique values and column order) ---
unique_pendidikan = ['SMA', 'SMK', 'D3', 'S1', 'SMP'] # From df_bersih['Pendidikan'].unique()
unique_jurusan = ['administrasi', 'teknik las', 'desain grafis', 'teknik listrik', 'otomotif'] # From df_bersih['Jurusan'].unique()
unique_jenis_kelamin = ['Laki-laki', 'Wanita'] # From df_bersih['Jenis_Kelamin'].unique()
unique_status_bekerja = ['Sudah Bekerja', 'Belum Bekerja'] # From df_bersih['Status_Bekerja'].unique()
X_train_cols = ['Usia', 'Durasi_Jam', 'Nilai_Ujian', 'Pendidikan', 'Jurusan', 'Jenis_Kelamin_Laki-laki', 'Jenis_Kelamin_Wanita', 'Status_Bekerja_Belum Bekerja', 'Status_Bekerja_Sudah Bekerja'] # From X_train.columns

# --- 3. Preprocessing Function (Mirrors training pipeline) ---
def preprocess_input(input_data, scaler, X_train_cols,
                     unique_pendidikan, unique_jurusan,
                     unique_jenis_kelamin, unique_status_bekerja):

    df_dummy = pd.DataFrame([input_data])

    # Initialize LabelEncoders and fit with known unique values
    le_pendidikan = LabelEncoder()
    le_pendidikan.fit(unique_pendidikan)

    le_jurusan = LabelEncoder()
    le_jurusan.fit(unique_jurusan)

    # Apply Label Encoding
    df_dummy['Pendidikan'] = le_pendidikan.transform(df_dummy['Pendidikan'])
    df_dummy['Jurusan'] = le_jurusan.transform(df_dummy['Jurusan'])

    # One-Hot Encoding for 'Jenis_Kelamin' and 'Status_Bekerja'
    # Create a temporary DataFrame for one-hot encoding categorical features
    df_onehot_temp = pd.DataFrame(index=df_dummy.index)
    
    for gender_val in unique_jenis_kelamin:
        col_name = f'Jenis_Kelamin_{gender_val}'
        df_onehot_temp[col_name] = (df_dummy['Jenis_Kelamin'] == gender_val).astype(int)

    for status_val in unique_status_bekerja:
        col_name = f'Status_Bekerja_{status_val}'
        df_onehot_temp[col_name] = (df_dummy['Status_Bekerja'] == status_val).astype(int)

    # Drop original categorical columns from df_dummy
    df_dummy = df_dummy.drop(columns=['Jenis_Kelamin', 'Status_Bekerja'])

    # Concatenate numerical, label-encoded, and one-hot encoded features
    temp_processed_df = pd.concat([
        df_dummy.drop(columns=['Pendidikan', 'Jurusan']), # Numerical columns
        df_dummy[['Pendidikan', 'Jurusan']],             # Label-encoded columns
        df_onehot_temp
    ], axis=1)

    # Reindex columns to match X_train_cols order and add missing one-hot columns with 0
    final_processed_df = pd.DataFrame(columns=X_train_cols, index=temp_processed_df.index)
    for col in X_train_cols:
        if col in temp_processed_df.columns:
            final_processed_df[col] = temp_processed_df[col]
        else:
            final_processed_df[col] = 0 # Fill with 0 for missing one-hot columns
    
    # Ensure data types are consistent before scaling
    final_processed_df = final_processed_df.infer_objects(copy=False) # Infer correct types
    
    # Convert any remaining objects (if any) to numeric, coercing errors
    for col in final_processed_df.columns:
        final_processed_df[col] = pd.to_numeric(final_processed_df[col], errors='coerce').fillna(0)


    # Scale numerical features using the loaded scaler
    scaled_data = scaler.transform(final_processed_df)

    return scaled_data

# --- 4. Streamlit UI ---
st.set_page_config(layout="wide")
st.title('Prediksi Gaji Pertama Peserta Pelatihan Vokasi')
st.markdown("Aplikasi ini memprediksi estimasi gaji pertama peserta pelatihan vokasi berdasarkan beberapa fitur.")

st.header("Input Data Peserta")

col1, col2 = st.columns(2)

with col1:
    usia = st.slider('Usia', min_value=18, max_value=60, value=30, help="Usia peserta pelatihan.")
    durasi_jam = st.slider('Durasi Jam Pelatihan', min_value=10, max_value=100, value=60, help="Total durasi pelatihan dalam jam.")
    nilai_ujian = st.slider('Nilai Ujian', min_value=0.0, max_value=100.0, value=85.0, step=0.1, help="Nilai akhir ujian pelatihan.")
    pendidikan = st.selectbox('Pendidikan', options=sorted(unique_pendidikan), help="Jenjang pendidikan terakhir peserta.")

with col2:
    jurusan = st.selectbox('Jurusan', options=sorted(unique_jurusan), help="Jurusan atau bidang pelatihan yang diambil.")
    jenis_kelamin = st.selectbox('Jenis Kelamin', options=sorted(unique_jenis_kelamin), help="Jenis kelamin peserta.")
    status_bekerja = st.selectbox('Status Bekerja', options=sorted(unique_status_bekerja), help="Status pekerjaan peserta setelah pelatihan.")


# Prediction button
st.markdown("--- ")
if st.button('Prediksi Gaji', help="Klik untuk mendapatkan prediksi gaji pertama."):
    input_data = {
        'Usia': usia,
        'Pendidikan': pendidikan,
        'Jurusan': jurusan,
        'Durasi_Jam': durasi_jam,
        'Nilai_Ujian': nilai_ujian,
        'Jenis_Kelamin': jenis_kelamin,
        'Status_Bekerja': status_bekerja
    }

    processed_input = preprocess_input(input_data, scaler, X_train_cols,
                                       unique_pendidikan, unique_jurusan,
                                       unique_jenis_kelamin, unique_status_bekerja)

    prediction = model.predict(processed_input)

    st.success(f"### Prediksi Gaji Pertama: {prediction[0]:.2f} Juta Rupiah")
    st.info("Prediksi ini adalah estimasi berdasarkan model yang telah dilatih dan mungkin bervariasi dari kondisi aktual.")
