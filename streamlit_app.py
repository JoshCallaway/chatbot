import streamlit as st
import joblib
import pandas as pd
import numpy as np
import statsmodels.api as sm

# --- Load your trained logistic regression model ---
model = joblib.load("c_t_lr_model.joblib")  # replace with actual filename

# Mapping of original variables and their dropdown options (labels + choices)
name_change = {
    'AInjAge': ['Age at Injury'], 
    'ASex': ['Sex', (1, 'Male'), (2, 'Female')], 
    'ARace': ['Race', (1, 'White'), (2, 'Black'), (3, 'American Indian'), (4, 'Asian'), (5, 'Other Race/Multiracial')],
    'AHispnic': ['Hispanic Origin', (0, 'Not of Hispanic Origin'), (1, 'Hispanic or Latin Origin'), (7, 'Declined/Does Not Know')],
    'AMarStIj': ['Marital Status', (1, 'Never Married (Single)'), (2, 'Married'), (3, 'Divorced'), (4, 'Seperated'), (5, 'Widowed'), (6, 'Other, unclassified'), (7, 'Living with Significant Other, Partner, Unmarried Couple')], 
    'AASAImDs': ['ASIA/Frankel Impairment', (5, 'ASIA A / Frankel Grade A'), (1, 'ASIA B / Frankel Grade B'), (2, 'ASIA C / Frankel Grade C'), (3, 'ASIA D / Frankel Grade D'), (4, 'ASIA E / Frankel Grade E')],
    'ANCatDis': ['Category of Neurological Impairment', (1, 'Normal Neurologic'), (2, 'Normal Neurologic, Minimal Neurologic Deficit'), (4, 'Paraplegia, incomplete'), (5, 'Paraplegia, complete'), (3, 'Paraplegia, minimal deficit'), (7, 'Tetraplegia, incomplete'), (8, 'Tetraplegia, complete'), (6, 'Tetraplegia, minimal deficit')],
    'APNFDisL': ['Level of Preserved Neurologic Function at Discharge', (0, 'C01'), (1, 'C02'), (2, 'C03'), (3, 'C04'), (4, 'C05'), (5, 'C06'), (6, 'C07'), (7, 'C08'), (8, 'Cervical, Unknown Level'), (9, 'T01'), (10, 'T02'), (11, 'T03'), (12, 'T04'), (13, 'T05'), (14, 'T06'), (15, 'T07'), (16, 'T08'), (17, 'T09'), (18, 'T10'), (19, 'T11'), (20, 'T12'), (21, 'Thoracic, Unknown Level'), (22, 'L01'), (23, 'L02'), (24, 'L03'), (25, 'L04'), (26, 'L05'), (27, 'Lumbar, Unknown Level'), (28, 'S01'), (29, 'S02'), (30, 'S03'), (31, 'S04'), (32, 'S05'), (33, 'Sacral, Unknown Level'), (34, 'Normal neurologic (INT)')],
    'ANurLvlD': ['Neurologic Level of Injury at Discharge', (0, 'C01'), (1, 'C02'), (2, 'C03'), (3, 'C04'), (4, 'C05'), (5, 'C06'), (6, 'C07'), (7, 'C08'), (8, 'Cervical, Unknown Level'), (9, 'T01'), (10, 'T02'), (11, 'T03'), (12, 'T04'), (13, 'T05'), (14, 'T06'), (15, 'T07'), (16, 'T08'), (17, 'T09'), (18, 'T10'), (19, 'T11'), (20, 'T12'), (21, 'Thoracic, Unknown Level'), (22, 'L01'), (23, 'L02'), (24, 'L03'), (25, 'L04'), (26, 'L05'), (27, 'Lumbar, Unknown Level'), (28, 'S01'), (29, 'S02'), (30, 'S03'), (31, 'S04'), (32, 'S05'), (33, 'Sacral, Unknown Level'), (34, 'Normal neurologic (INT)')],
    'AASATotD': ['ASIA Motor Index Score'],
    'ASLDisRt': ['Sensory Level at Discharge, Right', (0, 'C01'), (1, 'C02'), (2, 'C03'), (3, 'C04'), (4, 'C05'), (5, 'C06'), (6, 'C07'), (7, 'C08'), (8, 'Cervical, Unknown Level'), (9, 'T01'), (10, 'T02'), (11, 'T03'), (12, 'T04'), (13, 'T05'), (14, 'T06'), (15, 'T07'), (16, 'T08'), (17, 'T09'), (18, 'T10'), (19, 'T11'), (20, 'T12'), (21, 'Thoracic, Unknown Level'), (22, 'L01'), (23, 'L02'), (24, 'L03'), (25, 'L04'), (26, 'L05'), (27, 'Lumbar, Unknown Level'), (28, 'S01'), (29, 'S02'), (30, 'S03'), (31, 'S04'), (32, 'S05'), (33, 'Sacral, Unknown Level'), (34, 'Normal neurologic (INT)')],
    'ASLDisLf': ['Sensory Level at Discharge, Left', (0, 'C01'), (1, 'C02'), (2, 'C03'), (3, 'C04'), (4, 'C05'), (5, 'C06'), (6, 'C07'), (7, 'C08'), (8, 'Cervical, Unknown Level'), (9, 'T01'), (10, 'T02'), (11, 'T03'), (12, 'T04'), (13, 'T05'), (14, 'T06'), (15, 'T07'), (16, 'T08'), (17, 'T09'), (18, 'T10'), (19, 'T11'), (20, 'T12'), (21, 'Thoracic, Unknown Level'), (22, 'L01'), (23, 'L02'), (24, 'L03'), (25, 'L04'), (26, 'L05'), (27, 'Lumbar, Unknown Level'), (28, 'S01'), (29, 'S02'), (30, 'S03'), (31, 'S04'), (32, 'S05'), (33, 'Sacral, Unknown Level'), (34, 'Normal neurologic (INT)')],
    'AMLDisRt': ['Motor Level at Discharge, Right', (0, 'C01'), (1, 'C02'), (2, 'C03'), (3, 'C04'), (4, 'C05'), (5, 'C06'), (6, 'C07'), (7, 'C08'), (8, 'Cervical, Unknown Level'), (9, 'T01'), (10, 'T02'), (11, 'T03'), (12, 'T04'), (13, 'T05'), (14, 'T06'), (15, 'T07'), (16, 'T08'), (17, 'T09'), (18, 'T10'), (19, 'T11'), (20, 'T12'), (21, 'Thoracic, Unknown Level'), (22, 'L01'), (23, 'L02'), (24, 'L03'), (25, 'L04'), (26, 'L05'), (27, 'Lumbar, Unknown Level'), (28, 'S01'), (29, 'S02'), (30, 'S03'), (31, 'S04'), (32, 'S05'), (33, 'Sacral, Unknown Level'), (34, 'Normal neurologic (INT)')],
    'AMLDisLf': ['Motor Level at Discharge, Left', (0, 'C01'), (1, 'C02'), (2, 'C03'), (3, 'C04'), (4, 'C05'), (5, 'C06'), (6, 'C07'), (7, 'C08'), (8, 'Cervical, Unknown Level'), (9, 'T01'), (10, 'T02'), (11, 'T03'), (12, 'T04'), (13, 'T05'), (14, 'T06'), (15, 'T07'), (16, 'T08'), (17, 'T09'), (18, 'T10'), (19, 'T11'), (20, 'T12'), (21, 'Thoracic, Unknown Level'), (22, 'L01'), (23, 'L02'), (24, 'L03'), (25, 'L04'), (26, 'L05'), (27, 'Lumbar, Unknown Level'), (28, 'S01'), (29, 'S02'), (30, 'S03'), (31, 'S04'), (32, 'S05'), (33, 'Sacral, Unknown Level'), (34, 'Normal neurologic (INT)')],
    'APResDis_grouped': ['Place of Residence', ('Private/Home', 'Private/Home'), ('Institutional', 'Institutional'), ('Other', 'Other')],
    'ABdMMDis_grouped': ['Method of Bladder Management', ('No device', 'No device'), ('Indwelling', 'Indwelling'), ('Intermittent', 'Intermittent'), ('Suprapubic', 'Suprapubic'), ('Other/Unknown', 'Other/Unknown')],
    'AJobCnCd_grouped': ['Job Category', ('Healthcare/Professional', 'Healthcare/Professional'), ('Office/Admin', 'Office/Admin'), ('Professional/Skilled', 'Professional/Skilled'), ('Service/Manual', 'Service/Manual'), ('Skilled Labor', 'Skilled Labor')],
    'ATrmEtio_grouped': ['Traumatic Etiology', ('Motor vehicle', 'Motor vehicle'), ('Medical', 'Medical'), ('Sports', 'Sports'), ('Violence', 'Violence'),  ('Other/Unknown', 'Other/Unknown')]
}

# Group original variables before one-hot encoding
original_variables = [
    'AInjAge', 'ASex', 'ARace', 'AHispnic', 'AMarStIj',
    'APResDis_grouped', 'ABdMMDis_grouped', 'AJobCnCd_grouped', 'ATrmEtio_grouped',
    'AASAImDs', 'ANCatDis',
    'APNFDisL', 'ANurLvlD',
    'AASATotD', 'ASLDisRt', 'ASLDisLf', 'AMLDisRt', 'AMLDisLf'
]

# Map original variable name to one-hot encoded features that your model uses
one_hot_features_map = {
    'ARace': ['ARace_1', 'ARace_2', 'ARace_3', 'ARace_4', 'ARace_5'],  # Assuming 1-based for all races; adjust if needed
    'APResDis_grouped': ['APResDis_grouped_Private/Home', 'APResDis_grouped_Institutional', 'APResDis_grouped_Other'],
    'ABdMMDis_grouped': ['ABdMMDis_grouped_No device', 'ABdMMDis_grouped_Indwelling', 'ABdMMDis_grouped_Intermittent', 'ABdMMDis_grouped_Suprapubic', 'ABdMMDis_grouped_Other/Unknown'],
    'AJobCnCd_grouped': ['AJobCnCd_grouped_Healthcare/Professional', 'AJobCnCd_grouped_Office/Admin', 'AJobCnCd_grouped_Professional/Skilled', 'AJobCnCd_grouped_Service/Manual', 'AJobCnCd_grouped_Skilled Labor'],
    'ATrmEtio_grouped': ['ATrmEtio_grouped_Motor vehicle', 'ATrmEtio_grouped_Medical', 'ATrmEtio_grouped_Sports', 'ATrmEtio_grouped_Violence',  'ATrmEtio_grouped_Other/Unknown']
}

st.title("Return to Work Prediction After Cervicothoracic Spinal Cord Injury")
st.markdown(
    """
    <p style="font-size:16px; font-weight:normal;">
    This tool provides a <b>clinical support estimate</b> of the likelihood of returning to employment one year post-injury based on patient and injury characteristics.
    It is intended to aid clinicians in rehabilitation planning and is not a substitute for professional judgment or diagnosis.
    </p>
    """, 
    unsafe_allow_html=True
)

st.markdown("### Input patient and injury characteristics below:")
st.write(
    "<span style='font-size:12px; color:gray;'>⚠️ Please enter consistent clinical data. For example, ASIA impairment grade and ASIA  motor index score should match logically. Mismatched inputs may affect prediction reliability. </span>", 
    unsafe_allow_html=True
)

user_input = {}
default_values = {
    'AInjAge': 45,  # mid-age adult
    'ASex': 'Female',
    'ARace': 'Black',
    'AHispnic': 'Hispanic or Latin Origin',
    'AMarStIj': 'Widowed',
    'AASAImDs': 'ASIA B / Frankel Grade B',
    'ANCatDis': 'Tetraplegia, incomplete',
    'APNFDisL': 'C05',
    'ANurLvlD': 'C05',
    'AASATotD': 10,  # numeric default for motor index
    'ASLDisRt': 'C05',
    'ASLDisLf': 'C05',
    'AMLDisRt': 'C05',
    'AMLDisLf': 'C05',
    'APResDis_grouped': 'Institutional',
    'ABdMMDis_grouped': 'Indwelling',
    'AJobCnCd_grouped': 'Service/Manual',
    'ATrmEtio_grouped': 'Violence'
}

for var in original_variables:
    label = name_change[var][0]
    options = name_change[var][1:]  # list of tuples

    if len(options) > 0:
        display_options = [opt[1] for opt in options]

        # Get the default label for this variable
        default_label = default_values.get(var)

        # If default label is None or not found in options, select first option (index=0)
        if default_label is not None and default_label in display_options:
            default_index = display_options.index(default_label)
        else:
            default_index = 0

        selected_display = st.selectbox(label, display_options, index=default_index)

        # Map selected display back to code
        selected_code = None
        for code, name in options:
            if name == selected_display:
                selected_code = code
                break

        user_input[var] = selected_code

    else:
        # For numeric inputs, get default value if exists, else 0
        default_num = default_values.get(var, 0)
        user_input[var] = st.number_input(label, value=default_num, step=1, format="%d")


# Now build the final input DataFrame with all one-hot features set
# Start with zeros for all features your model expects
all_features = [
    'AInjAge', 'ASex', 'AMarStIj', 'AHispnic', 'AASAImDs', 'ANCatDis', 'APNFDisL', 'ANurLvlD',
    'AASATotD', 'ASLDisRt', 'ASLDisLf', 'AMLDisRt', 'AMLDisLf',
    # one hot features:
    'ARace_2', 'ARace_3', 'ARace_4', 'ARace_5',
    'APResDis_grouped_Institutional', 'APResDis_grouped_Other',
    'ABdMMDis_grouped_Indwelling', 'ABdMMDis_grouped_Intermittent',
    'ABdMMDis_grouped_No device', 'ABdMMDis_grouped_Other/Unknown',
    'ABdMMDis_grouped_Suprapubic', 'AJobCnCd_grouped_Healthcare/Professional',
    'AJobCnCd_grouped_Office/Admin', 'AJobCnCd_grouped_Professional/Skilled',
    'AJobCnCd_grouped_Service/Manual', 'AJobCnCd_grouped_Skilled Labor',
    'ATrmEtio_grouped_Medical', 'ATrmEtio_grouped_Motor vehicle',
    'ATrmEtio_grouped_Other/Unknown', 'ATrmEtio_grouped_Sports',
    'ATrmEtio_grouped_Violence'
]

final_input = {feat: 0 for feat in all_features}

# Add numeric / single categorical features directly (all except those that are one-hot encoded groups)
for feat in all_features:
    if feat in user_input:
        # direct numeric or categorical (not one hot)
        final_input[feat] = user_input[feat]

# Encode one-hot groups from user_input
# ARace example: races 1 to 5, model uses ARace_2..5 as columns, assuming ARace_1 is baseline and dropped
# So if ARace == 1, none of ARace_2..5 is set; if ARace == 2, ARace_2=1 etc.

# ARace one-hot encoding
arace_val = user_input.get('ARace')
if arace_val is not None:
    # baseline is ARace_1, so no column set for ARace_1
    if arace_val != 1:
        col = f'ARace_{arace_val}'
        if col in final_input:
            final_input[col] = 1

# APResDis_grouped encoding
apres_val = user_input.get('APResDis_grouped')
if apres_val is not None:
    # baseline probably 'Private/Home' (no column)
    if apres_val != 'Private/Home':
        col = f'APResDis_grouped_{apres_val}'
        if col in final_input:
            final_input[col] = 1

# ABdMMDis_grouped encoding
abdmm_val = user_input.get('ABdMMDis_grouped')
if abdmm_val is not None:
    # baseline probably 'No device' (no column)
    if abdmm_val != 'No device':
        col = f'ABdMMDis_grouped_{abdmm_val}'
        if col in final_input:
            final_input[col] = 1

# AJobCnCd_grouped encoding
ajob_val = user_input.get('AJobCnCd_grouped')
if ajob_val is not None:
    # baseline maybe first group - no column set
    if ajob_val != 'Healthcare/Professional':
        col = f'AJobCnCd_grouped_{ajob_val}'
        if col in final_input:
            final_input[col] = 1

# ATrmEtio_grouped encoding
atrm_val = user_input.get('ATrmEtio_grouped')
if atrm_val is not None:
    # baseline maybe 'Medical' - no column set
    if atrm_val != 'Medical':
        col = f'ATrmEtio_grouped_{atrm_val}'
        if col in final_input:
            final_input[col] = 1


# Convert to DataFrame
input_df = pd.DataFrame([final_input])

# Your existing predict button and output
if st.button("Predict"):
    input_df_with_const = sm.add_constant(input_df, has_constant='add')  # add intercept column
    pred_prob = model.predict(input_df_with_const)[0]  # predicted probability
    pred_prob = 1 - pred_prob
    pred_class = int(pred_prob >= 0.5135)       # optimal threshold for LR (max Youden's J)

    st.subheader("Prediction Results")
    st.write(f"**Estimated Percent Change of Return to Work 1 Year After Injury:** {pred_prob * 100:.1f}%")
    st.write(f"**Predicted Class:** {'Employed' if pred_class == 1 else 'Unemployed'}")

# Add disclaimer and links here (outside the if block, so always shown)

st.markdown(
    """
    ---
    Based on data provided by the **National Spinal Cord Injury Statistical Center (NSCISC)**  
    
    Developed by Josh Callaway in collaboration with the *Spine Research Group* at **UC Davis Medical Center**, led by Dr. Hai Le  

    **Josh Callaway:** [LinkedIn](https://www.linkedin.com/in/josh-callaway-a79661226/) | Research Profile: [ResearchGate](https://www.researchgate.net/profile/Josh-Callaway-2?ev=hdr_xprf)  
    **Dr. Hai Le:** [LinkedIn](https://www.linkedin.com/in/hai-le-866b76b0/) | Research Profile: [ResearchGate](https://www.researchgate.net/profile/Hai-Le-42/research)  
    """
)

st.markdown(
    """
    <hr>
    <p style="font-size:10px; color:gray; text-align:center;">
    &copy; 2025 Josh Callaway, in collaboration with UC Davis Medical Center. All rights reserved.  

    </p>
    """,
    unsafe_allow_html=True
)

# # Better Copyright link (for future use)
# st.markdown(
#     """
#     <hr>
#     <p style="font-size:10px; color:gray; text-align:center;">
#     &copy; 2025 Josh Callaway. All rights reserved.  
#     Visit <a href="https://www.yourwebsite.com" target="_blank" style="color:gray; text-decoration:none;">yourwebsite.com</a>
#     </p>
#     """,
#     unsafe_allow_html=True
# )


# st.markdown(
#     """
#     <hr>
#     <p style="font-size:10px; color:gray; text-align:center;">
#     Last updated: 8/11/2025
#     </p>
#     """,
#     unsafe_allow_html=True
# )

