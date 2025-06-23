import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle
import os

# Set page config
st.set_page_config(
    page_title="Prediksi Diabetes",
    page_icon="🩺",
    layout="wide"
)

# Load saved model
@st.cache_resource
def load_saved_model():
    """Load model yang sudah disimpan"""
    try:
        # Load model (gunakan joblib karena lebih efisien untuk sklearn)
        model = pickle.load(open('diabetes_model.pkl', 'rb'))
        
        # Load model info
        with open('model_info.pkl', 'rb') as file:
            model_info = pickle.load(file)
        
        return model, model_info
    
    except FileNotFoundError as e:
        st.error(f"""
         **Model file tidak ditemukan!**
        
        Pastikan Anda sudah menjalankan script training terlebih dahulu:
        ```python
        python train_and_save_model.py
        ```
        
        File yang dibutuhkan:
        - diabetes_model.pkl
        - model_info.pkl
        """)
        return None, None

@st.cache_data
def load_dataset():
    """Load dataset untuk informasi saja"""
    try:
        return pd.read_csv('Diabetes.csv')
    except FileNotFoundError:
        st.warning("Dataset tidak ditemukan untuk ditampilkan.")
        return None

# Main app
def main():
    st.title("🩺 Aplikasi Prediksi Diabetes")
    st.markdown("---")
    
    # Load model and data
    model, model_info = load_saved_model()
    dataset = load_dataset()
    
    if model is None:
        st.stop()
    
    # Sidebar for navigation
    st.sidebar.title("Navigasi")
    page = st.sidebar.selectbox("Pilih Halaman", ["Prediksi", "Informasi Model", "Dataset"])
    
    if page == "Prediksi":
        prediction_page(model)
    elif page == "Informasi Model":
        model_info_page(model_info)
    else:
        dataset_page(dataset)

def prediction_page(model):
    st.header("Prediksi Diabetes")
    st.write("Masukkan data pasien untuk memprediksi kemungkinan diabetes:")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input(
            "Jumlah Kehamilan", 
            min_value=0, max_value=20, 
            help="Jumlah kehamilan yang pernah dialami"
        )
        
        glucose = st.number_input(
            "Kadar Glukosa", 
            min_value=0, max_value=200, 
            help="Konsentrasi glukosa plasma dalam tes toleransi glukosa oral 2 jam"
        )
        
        blood_pressure = st.number_input(
            "Tekanan Darah", 
            min_value=0, max_value=200, 
            help="Tekanan darah diastolik (mm Hg)"
        )
        
        skin_thickness = st.number_input(
            "Ketebalan Kulit", 
            min_value=0, max_value=100, 
            help="Ketebalan lipatan kulit trisep (mm)"
        )
    
    with col2:
        insulin = st.number_input(
            "Insulin", 
            min_value=0, max_value=900,
            help="Insulin serum 2 jam (mu U/ml)"
        )
        
        bmi = st.number_input(
            "BMI", 
            min_value=0.0, max_value=70.0, value=0.0, format="%.1f",
            help="Indeks massa tubuh (berat dalam kg/(tinggi dalam m)^2)"
        )
        
        diabetes_pedigree = st.number_input(
            "Diabetes Pedigree Function", 
            min_value=0.0, max_value=3.0, value=0.0, format="%.3f",
            help="Fungsi silsilah diabetes"
        )
        
        age = st.number_input(
            "Umur", 
            min_value=1, max_value=120, 
            help="Umur dalam tahun"
        )
    
    # Prediction button
    if st.button("Prediksi Diabetes", type="primary"):
        # Prepare input data
        input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, 
                              insulin, bmi, diabetes_pedigree, age]).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Display result
        st.markdown("---")
        if prediction[0] == 0:
            st.success("**Hasil Prediksi: TIDAK DIABETES**")
            st.balloons()
        else:
            st.error("**Hasil Prediksi: DIABETES**")
            st.warning("Disarankan untuk berkonsultasi dengan dokter untuk pemeriksaan lebih lanjut.")
        
        # Show input summary
        with st.expander("Ringkasan Data Input"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Jumlah Kehamilan:** {pregnancies}")
                st.write(f"**Kadar Glukosa:** {glucose}")
                st.write(f"**Tekanan Darah:** {blood_pressure}")
                st.write(f"**Ketebalan Kulit:** {skin_thickness}")
            with col2:
                st.write(f"**Insulin:** {insulin}")
                st.write(f"**BMI:** {bmi}")
                st.write(f"**Diabetes Pedigree:** {diabetes_pedigree}")
                st.write(f"**Umur:** {age}")

def model_info_page(model_info):
    st.header("Informasi Model")
    
    if model_info is None:
        st.error("Informasi model tidak tersedia.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        train_acc = model_info.get('training_accuracy', 0)
        st.metric("Akurasi Training", f"{train_acc:.4f}", f"{train_acc*100:.2f}%")
    
    with col2:
        test_acc = model_info.get('test_accuracy', 0)
        st.metric("Akurasi Testing", f"{test_acc:.4f}", f"{test_acc*100:.2f}%")
    
    st.markdown("---")
    
    st.subheader("Spesifikasi Model")
    st.write(f"""
    - **Algoritma:** {model_info.get('model_type', 'SVM')}
    - **Kernel:** {model_info.get('kernel', 'Linear')}
    - **Train/Test Split:** {int((1-model_info.get('test_size', 0.2))*100)}%/{int(model_info.get('test_size', 0.2)*100)}%
    - **Stratifikasi:** Ya (berdasarkan target variable)
    - **Random State:** {model_info.get('random_state', 2)}
    """)
    
    st.subheader("Fitur Input")
    features_info = {
        "Pregnancies": "Jumlah kehamilan",
        "Glucose": "Konsentrasi glukosa plasma dalam tes toleransi glukosa oral 2 jam",
        "BloodPressure": "Tekanan darah diastolik (mm Hg)",
        "SkinThickness": "Ketebalan lipatan kulit trisep (mm)",
        "Insulin": "Insulin serum 2 jam (mu U/ml)",
        "BMI": "Indeks massa tubuh (berat dalam kg/(tinggi dalam m)^2)",
        "DiabetesPedigreeFunction": "Fungsi silsilah diabetes",
        "Age": "Umur (tahun)"
    }
    
    for feature, description in features_info.items():
        st.write(f"**{feature}:** {description}")

def dataset_page(dataset):
    st.header("Informasi Dataset")
    
    if dataset is None:
        st.warning("Dataset tidak dapat ditampilkan.")
        return
    
    # Dataset overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Sampel", dataset.shape[0])
    
    with col2:
        st.metric("Jumlah Fitur", dataset.shape[1]-1)
    
    with col3:
        diabetes_count = dataset['Outcome'].sum()
        st.metric("Pasien Diabetes", diabetes_count)
    
    st.markdown("---")
    
    # Show sample data
    st.subheader("Sample Data")
    st.dataframe(dataset.head(10))
    
    # Dataset statistics
    st.subheader("Statistik Dataset")
    st.dataframe(dataset.describe())
    
    # Outcome distribution
    st.subheader("Distribusi Target")
    outcome_counts = dataset['Outcome'].value_counts()
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Tidak Diabetes (0):**", outcome_counts[0])
        st.write("**Diabetes (1):**", outcome_counts[1])
    
    with col2:
        st.write("**Persentase Tidak Diabetes:**", f"{outcome_counts[0]/len(dataset)*100:.1f}%")
        st.write("**Persentase Diabetes:**", f"{outcome_counts[1]/len(dataset)*100:.1f}%")

if __name__ == "__main__":
    main()