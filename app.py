import streamlit as st
import pickle
import numpy as np

# Import the model
pipe = pickle.load(open('pipe.pkl', 'rb'), encoding='latin1')
pipe1 = pickle.load(open('pipe1.pkl', 'rb'), encoding='latin1')
df = pickle.load(open('df.pkl', 'rb'), encoding='latin1')

# Set page title and background color
st.set_page_config(page_title="Laptop Predictor", page_icon="💻", layout="centered", initial_sidebar_state="expanded")
st.markdown('<style>body{background-color: #f5f5f5;}</style>', unsafe_allow_html=True)

# Page title
st.title("Laptop Predictor")

# Brand
company = st.selectbox('Brand', df['Brand'].unique())

# Type of cpu
typecpu = st.selectbox('CPU', df['cpu_brand'].unique())

# RAM
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# Weight
weight = st.number_input('Weight of the Laptop')

# storage
storage_size = st.number_input('Storage size')

# hz
hz = st.selectbox('Refresh rate', df['displ_rate'].unique())

# battery
battery = st.number_input('Battery')

# ram_upgradable
ram_upgradable = st.selectbox('Ram Upgradable', ['No', 'Yes'])

# screen tech
screen_tech = st.selectbox('Screen Technology', df['screen_technology'].unique())


#storage_extra_slot
storage_extra_slot = st.selectbox('Storage extra slot', ['No', 'Yes'])

#gpu brand
gpu = st.selectbox('GPU brand', df['GPU_brand'].unique())

# Screen size
screen_size = st.number_input('Screen Size')

# Resolution
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])

# OS
os = st.selectbox('OS', df['OS'].unique())

# commont count
cmt = st.number_input('Comment count')

if st.button('Predict Price'):
    # Convert input values to appropriate format
    if ram_upgradable == 'Yes':
        ram_upgradable = 1
    else:
        ram_upgradable = 0

    if storage_extra_slot == 'Yes':
        storage_extra_slot = 1
    else:
        storage_extra_slot = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

    # Create query array
    query = np.array([cmt, company, typecpu, ram, storage_size, hz, battery, os, weight, ram_upgradable, screen_tech,storage_extra_slot, gpu,ppi])
    query = query.reshape(1, 14)

    # Predict price
    predicted_price = int(np.exp(pipe1.predict(query)[0]))

    # Show predicted price
    st.success("The predicted price of this configuration is $" + str(predicted_price))
