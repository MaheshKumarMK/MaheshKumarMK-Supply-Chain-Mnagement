import streamlit as st
import dill
import numpy as np
import pandas as pd


def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = dill.load(file)
    return data

data = load_model()

model = data["model"]
pipe = data["pipeline"]

def show_predict_page():
    st.set_page_config(page_title="WARE HOUSE :blue[_PRODUCT WEIGHT_] PREDICTION",layout='wide', initial_sidebar_state="expanded")
    st.title("WARE HOUSE :blue[_PRODUCT WEIGHT_] PREDICTION")
    
    st.write("""### We need some information to predict the product weight""")

    location_type = ("Urban", "Rural")
    wh_capacity_size = ("Small", "Large", "Mid")
    zone = ("West", "North", "East", "South")
    wh_regional_zone = ("Zone 1","Zone 2","Zone 3","Zone 4","Zone 5", "Zone 6")
    wh_owner_type = ("Rented", "Company Owned")
    approved_wh_govt_certificate = ("A", "A+", "B","C","B+")

    location_type = st.selectbox("Location of warehouse-like in city or village", location_type)
    wh_capacity_size = st.selectbox("Storage capacity size of the warehouse", wh_capacity_size)
    zone = st.selectbox("Zone of the warehouse", zone)
    wh_regional_zone = st.selectbox("Regional zone of the warehouse under each zone", wh_regional_zone)
    num_refill_req_l3m = st.slider("Number of times refilling has been done in last 3 months",min_value=0, max_value=20)
    transport_issue_l1y = st.slider("Any transport issue (Type 0 for 'No', 1 for 'Yes')",min_value=0, max_value=1)
    competitor_in_mkt=st.number_input("Number of instant noodles competitors in the market", value=0, step=1, format="%d")
    retail_shop_num = st.number_input("Number of the retail shop that sell the product under the warehouse area", value=0, step=1, format="%d")
    wh_owner_type = st.selectbox("Company is owning the warehouse or they have got the  warehouse on rent", wh_owner_type)
    distributor_num = st.number_input("The number of distributer works in between warehouse and retail shops", value=0, step=1, format="%d")
    flood_impacted = st.slider("Warehouse is in the Flood impacted area (Type 0 for 'No', 1 for 'Yes')",min_value=0, max_value=1)
    flood_proof = st.slider("AWarehouse is a flood-proof indicator (Type 0 for 'No', 1 for 'Yes')",min_value=0, max_value=1)
    electric_supply = st.slider("Warehouse have electric back up (Type 0 for 'No', 1 for 'Yes')",min_value=0, max_value=1)
    dist_from_hub = st.number_input("Distance between warehouse to the production hub in Kms", value=0, step=1, format="%d")
    workers_num = st.number_input("Number of workers working in the warehouse", value=0, step=1, format="%d")
    storage_issue_reported_l3m = st.number_input("Warehouse reported storage issue to corporate office in last 3 months", value=0, step=1, format="%d")
    temp_reg_mach = st.slider("Warehouse have temperature regulating machine indicator (Type 0 for 'No', 1 for 'Yes')",min_value=0, max_value=1)
    approved_wh_govt_certificate = st.selectbox("Kind of standard certificate issued to the warehouse from government", approved_wh_govt_certificate)
    wh_breakdown_l3m = st.slider("Number of time warehouse face a breakdown in last 3 months",min_value=0, max_value=20)
    govt_check_l3m = st.number_input("Number of time Officers visited the warehouse to check the quality in last 3 months", value=0, step=1, format="%d")

    ok = st.button("Calculate Product Weight")
    if ok:
        X = pd.DataFrame({
            "location_type":[location_type], 
            "wh_capacity_size":[wh_capacity_size], 
            "zone":[zone], 
            "wh_regional_zone":[wh_regional_zone],
            "num_refill_req_l3m":[num_refill_req_l3m],
            "transport_issue_l1y":[transport_issue_l1y],
            "competitor_in_mkt":[competitor_in_mkt],
            "retail_shop_num":[retail_shop_num],
            "wh_owner_type":[wh_owner_type],
            "distributor_num":[distributor_num],
            "flood_impacted":[flood_impacted],
            "flood_proof":[flood_proof],
            "electric_supply":[electric_supply],
            "dist_from_hub":[dist_from_hub],
            "workers_num":[workers_num],
            "storage_issue_reported_l3m":[storage_issue_reported_l3m],
            "temp_reg_mach":[temp_reg_mach],
            "approved_wh_govt_certificate":[approved_wh_govt_certificate],
            "wh_breakdown_l3m":[wh_breakdown_l3m],
            "govt_check_l3m":[govt_check_l3m]
            })
        X = pipe.transform(X)
        Product_weight = model.predict(X)
        st.subheader(f"The estimated product weight is {Product_weight[0]:.2f}kg")