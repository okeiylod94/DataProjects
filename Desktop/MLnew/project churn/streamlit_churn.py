import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Page config
st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📱",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    model = joblib.load("Top7_rf_model.pickle")
    return model

# Load model once
model = load_model()

# Title and description
st.title("📱 Telecom Churn Predictor")
st.markdown("**85% Accurate Model** - Enter customer usage data to predict churn risk")

# Sidebar for inputs
st.sidebar.header("Customer Usage Data")
st.sidebar.markdown("Enter the 7 key features:")

# Input widgets (same order as model training)
Day_Mins = st.sidebar.slider("Day Minutes", 0.0, 300.0, 150.0, 0.1)
Night_Charge = st.sidebar.slider("Night Charge ($)", 0.0, 20.0, 10.0, 0.1)
Day_Charge = st.sidebar.slider("Day Charge ($)", 0.0, 60.0, 30.0, 0.1)
Eve_Mins = st.sidebar.slider("Evening Minutes", 0.0, 200.0, 100.0, 0.1)
Eve_Charge = st.sidebar.slider("Evening Charge ($)", 0.0, 30.0, 15.0, 0.1)
Night_Calls = st.sidebar.slider("Night Calls", 0, 200, 80)
Vmail_Message = st.sidebar.slider("Voicemail Messages", 0, 50, 10)

# Main prediction column
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Input Summary")
    input_data = pd.DataFrame({
        'Feature': ['Day_Mins', 'Night_Charge', 'Day_Charge', 'Eve_Mins', 
                   'Eve_Charge', 'Night_Calls', 'Vmail_Message'],
        'Value': [Day_Mins, Night_Charge, Day_Charge, Eve_Mins, 
                 Eve_Charge, Night_Calls, Vmail_Message]
    })
    st.dataframe(input_data, use_container_width=True)

# Predict button
if st.sidebar.button("🔮 Predict Churn Risk", type="primary"):
    with col2:
        st.subheader("🎯 Prediction Results")
        
        # Create prediction DataFrame (EXACT model order)
        data = [[Day_Mins, Night_Charge, Day_Charge, Eve_Mins, 
                Eve_Charge, Night_Calls, Vmail_Message]]
        df = pd.DataFrame(data, columns=[
            "Day_Mins", "Night_Charge", "Day_Charge", "Eve_Mins",
            "Eve_Charge", "Night_Calls", "Vmail_Message"
        ])
        
        # Make prediction
        prediction = model.predict(df)[0]
        churn_prob = model.predict_proba(df)[0][1]
        confidence = max(model.predict_proba(df)[0])
        
        # Results
        if prediction == 1:
            st.error(f"🚨 **HIGH CHURN RISK**")
            st.metric("Churn Probability", f"{churn_prob:.1%}", delta="High Risk")
        else:
            st.success(f"✅ **LOW CHURN RISK**")
            st.metric("Churn Probability", f"{churn_prob:.1%}", delta="Low Risk")
        
        st.metric("Model Confidence", f"{confidence:.1%}")
        
        # Risk interpretation
        if churn_prob > 0.7:
            st.warning("💡 **Action:** Offer retention discount + night calling plan")
        elif churn_prob > 0.5:
            st.info("💡 **Action:** Monitor closely + send satisfaction survey")
        else:
            st.info("💡 **Action:** Continue normal service")

# Model info sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("**Model Performance**")
    st.success("✅ **85% Accurate**")
    st.info("🔍 **Top 7 Telecom Features**")
    st.markdown("""
    - Day_Mins (1st)
    - Night_Charge (2nd) 
    - Day_Charge (3rd)
    - Eve_Mins (4th)
    - Eve_Charge (5th)
    - Night_Calls (6th)
    - Vmail_Message (7th)
    """)
    
    st.markdown("---")
    st.markdown("[📖 Model Documentation](http://localhost:8000/docs)")
