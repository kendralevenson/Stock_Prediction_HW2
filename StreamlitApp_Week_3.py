import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath

import joblib
import tarfile
import tempfile

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer
from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import NumpyDeserializer

from sklearn.pipeline import Pipeline
import shap


# Setup & Path Configuration
warnings.simplefilter("ignore")

# Fix path for Streamlit Cloud (ensure 'src' is findable)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.feature_utils import extract_features

# Access the secrets
aws_id = st.secrets["aws_credentials"]["ASIARZCTM7SCDOUAW3IO"]
aws_secret = st.secrets["aws_credentials"]["TKBK3jQ8I5sVJmEJGeWNDWlxM8bRjpp2KdOqc0BY"]
aws_token = st.secrets["aws_credentials"]["IQoJb3JpZ2luX2VjELv//////////wEaCXVzLXdlc3QtMiJHMEUCIDUNb94n60g2uLP1Ohetm2ermPFxTVSkmyfKzNPQLvkKAiEA1reFK65jeLdZvcMZ/TC3LSC8jShqvWBthPorWREQJi4qsQIIhP//////////ARACGgwxMjI1ODE0ODI2MjgiDFggC60tXDQPqBCkeSqFAm4hi7lTKPRGogu/+kTODT56JXi67lafF+y5p10JbOMPJ5M1Yj9Adf/kjj6Jk/lgDgkR5l9aa02cBNoW3zO/X6NCiHHbyPtUXxomqSkfec31GCbP2s8tcrLJyPPaNX5CARJmvv4c9b0T/ak063x3ujg+bC2xB2YzDKZPBQn1ektzKXttjVFZDnu3qPnK8QV9A8fr2uQk1QDEMzyiCw6k2k65JpCwsbcFaJdcxT4JQgOjIrQkg9UTXGumQxr46Ps5lO0eo9xLtMGiATjTlMjccjwN2DXhA0AbgDjvkCymzA+0KH5VbXs63n7cu+N9008gqpY7y1usLeXrBijgKOYivJS+uIuNhzDlrt3MBjqdAZQcUBtfAQ+KhBHXMjkAK0PAMXT5iVtIbcaKbfS4o7jusaEDUHfPOlJp8bilnyk3+a3+2Jk1BBLrpEYjtJR6Zwp9cfSPn6ESQV8ApVTviZOO0zZBeSn+4ylllcGgzvaiTjk34Zkxxi+/Y+X9r5a38q4YaedKXU+S3OuAD3b5m6YrHlYagiauTH3/rV7dv68s3L7Mxn7GwYJ/jMkZ8jU="]
aws_bucket = st.secrets["aws_credentials"]["kendra-stock-app-2026"]
aws_endpoint = st.secrets["aws_credentials"]["https://s3.amazonaws.com"]

# AWS Session Management
@st.cache_resource # Use this to avoid downloading the file every time the page refreshes
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name='us-east-1'
    )

session = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

# Data & Model Configuration
df_features = extract_features()

MODEL_INFO = {
        "endpoint": aws_endpoint,
        "explainer": 'explainer.shap',
        "pipeline": 'finalized_model.tar.gz',
        "keys": ["AAPL", "MSFT", "DEXJPUS", "DEXUSUK", "SP500", "DJIA", "VIXCLS"],
        "inputs": [{"name": k, "type": "number", "min": -1.0, "max": 1.0, "default": 0.0, "step": 0.01} for k in ["AAPL", "MSFT", "DEXJPUS", "DEXUSUK", "SP500", "DJIA", "VIXCLS"]]
}

def load_pipeline(_session, bucket, key):
    s3_client = _session.client('s3')
    filename=MODEL_INFO["pipeline"]

    s3_client.download_file(
        Filename=filename, 
        Bucket=bucket, 
        Key= f"{key}/{os.path.basename(filename)}")
        # Extract the .joblib file from the .tar.gz
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        joblib_file = [f for f in tar.getnames() if f.endswith('.joblib')][0]

    # Load the full pipeline
    return joblib.load(f"{joblib_file}")

def load_shap_explainer(_session, bucket, key, local_path):
    s3_client = _session.client('s3')
    local_path = local_path

    # Only download if it doesn't exist locally to save time
    if not os.path.exists(local_path):
        s3_client.download_file(Filename=local_path, Bucket=bucket, Key=key)
        
    with open(local_path, "rb") as f:
        return shap.Explainer.load(f)

# Prediction Logic
def call_model_api(input_df):

    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer() 
    )

    try:
        raw_pred = predictor.predict(input_df)
        pred_val = pd.DataFrame(raw_pred).values[-1][0]
        return round(float(pred_val), 4), 200
    except Exception as e:
        return f"Error: {str(e)}", 500

# Local Explainability
def display_explanation(input_df, session, aws_bucket):
    explainer_name = MODEL_INFO["explainer"]
    explainer = load_shap_explainer(session, aws_bucket, posixpath.join('explainer', explainer_name),os.path.join(tempfile.gettempdir(), explainer_name))
    shap_values = explainer(input_df)
    st.subheader("🔍 Decision Transparency (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0], max_display=10)
    st.pyplot(fig)
    # top feature   
    top_feature = shap_values[0].feature_names[0]
    st.info(f"**Business Insight:** The most influential factor in this decision was **{top_feature}**.")

# Streamlit UI
st.set_page_config(page_title="ML Deployment", layout="wide")
st.title("👨‍💻 ML Deployment")

with st.form("pred_form"):
    st.subheader(f"Inputs")
    cols = st.columns(2)
    user_inputs = {}
    
    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp['name']] = st.number_input(
                inp['name'].replace('_', ' ').upper(),
                min_value=inp['min'], max_value=inp['max'], value=inp['default'], step=inp['step']
            )
    
    submitted = st.form_submit_button("Run Prediction")

if submitted:

    data_row = [user_inputs[k] for k in MODEL_INFO["keys"]]
    # Prepare data
    base_df = df_features
    input_df = pd.concat([base_df, pd.DataFrame([data_row], columns=base_df.columns)])
    
    res, status = call_model_api(input_df)
    if status == 200:
        st.metric("Prediction Result", res)
        display_explanation(input_df,session, aws_bucket)
    else:
        st.error(res)






