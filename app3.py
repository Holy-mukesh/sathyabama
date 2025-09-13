import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from streamlit_geolocation import streamlit_geolocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from googletrans import Translator
import datetime
import os

st.set_page_config(page_title="AI-Assisted Telemedicine", page_icon="🩺", layout="wide")

LOG_FILE = "user_daily_logs.csv"
PROGRESS_FILE = "disease_progress_logs.csv"

# --- Place this dictionary at the TOP before any function that uses it! ---
DISEASE_PROGRESS_QUESTIONS = {
    "Fungal infection": [
        "Is the affected area less itchy?",
        "Is there a reduction in redness?",
        "Has there been a decrease in scaling?",
        "Any new patches appeared?",
    ],
    "Common Cold": [
        "Is your nasal congestion improving?",
        "Are you coughing less?",
        "Do you feel less tired?",
        "Are you able to sleep better?"
    ],
    "Impetigo": [
        "Has crusting reduced?",
        "Is the area less painful?",
        "Has the redness subsided?",
        "Are you applying medication regularly?",
    ],
    "Dengue": [
        "Is your fever subsiding?",
        "Has joint pain decreased?",
        "Is your appetite normal?",
        "Are you experiencing any bleeding?",
    ],
    "Typhoid": [
        "Are you feeling less fatigued?",
        "Has abdominal pain reduced?",
        "Are you tolerating food?",
        "Is diarrhea improving?",
    ],
    "Chicken pox": [
        "Are new blisters forming?",
        "Is itching improving?",
        "Has fever resolved?",
        "Are scabs healing?",
    ],
    "Psoriasis": [
        "Is scaling improving?",
        "Has skin redness lessened?",
        "Are plaques shrinking?",
        "Is itch better controlled?",
    ],
    "Varicose Veins": [
        "Is leg pain reducing?",
        "Is swelling decreasing?",
        "Are there new skin color changes?",
        "Are you able to walk better?",
    ],
    "Jaundice": [
        "Is your skin/yellowing fading?",
        "Is your appetite improving?",
        "Is abdominal pain reducing?",
        "Are you feeling less fatigued?",
    ],
    "Malaria": [
        "Has your fever subsided?",
        "Are chills reducing?",
        "Do you feel less fatigued?",
        "Is abdominal discomfort gone?",
    ],
    "Bronchial Asthma": [
        "Are breathing difficulties reduced?",
        "Is your cough better?",
        "Are you using your inhaler less often?",
        "Do you experience chest tightness?",
    ],
    "Hypertension": [
        "Is your blood pressure within normal range?",
        "Any dizziness today?",
        "Are you compliant with medications?",
        "Any headaches today?",
    ],
    "Migraine": [
        "Has headache frequency reduced?",
        "Is pain intensity less?",
        "Are nausea/vomiting improving?",
        "Are lights/noises less bothersome?",
    ],
    "Cervical spondylosis": [
        "Is neck pain less severe?",
        "Have you noticed more flexibility?",
        "Is tingling in arms decreased?",
        "Any improvement in headaches?",
    ],
    "diabetes": [
        "Is your blood sugar controlled?",
        "Any dizziness or fainting spells?",
        "Any increase in thirst?",
        "Are wounds healing normally?",
    ],
    "drug reaction": [
        "Are skin symptoms improving?",
        "Is itchiness reduced?",
        "Any new symptoms appeared?",
        "Is medication stopped?",
    ],
    "peptic ulcer disease": [
        "Is abdominal pain improving?",
        "Any vomiting or bleeding?",
        "Is appetite normal?",
        "Are you sticking to prescribed medications?",
    ],
    "urinary tract infection": [
        "Is frequency of urination reducing?",
        "Is burning sensation improved?",
        "Any fever present?",
        "Any blood in urine?",
    ],
    "allergy": [
        "Are itching/allergy symptoms improving?",
        "Any swelling of lips/tongue?",
        "Are you using allergy medications?",
        "Any new triggers noticed?",
    ],
}
# -------------------------------------------------------------------------

def load_data():
    diseases_symptoms = pd.read_csv("Final_Augmented_dataset_Diseases_and_Symptoms.csv")
    symptom_disease = pd.read_csv("Symptom2Disease.csv")
    medication_db = pd.read_csv("disease_to_example_medications.csv")
    return diseases_symptoms, symptom_disease, medication_db

diseases_symptoms, symptom_disease, medication_db = load_data()
vectorizer = TfidfVectorizer()
X_symptoms = vectorizer.fit_transform(symptom_disease['symptoms'].astype(str))
translator = Translator()

def predict_diseases(text, top_n=3):
    sims = cosine_similarity(vectorizer.transform([text]), X_symptoms).flatten()
    idxs = sims.argsort()[::-1][:top_n]
    return [(symptom_disease.iloc[i]['disease'], round(sims[i]*100, 2)) for i in idxs]

def triage_level(conf):
    if conf >= 80: return "High"
    elif conf >= 50: return "Medium"
    return "Low"

def triage_msg(conf):
    lvl = triage_level(conf)
    return "🔥 Urgent medical attention recommended" if lvl == "High" else \
           "⚠️ Consult a doctor soon" if lvl == "Medium" else \
           "✅ Self-care and monitor symptoms"

def find_medications(disease):
    disease_norm = disease.strip().lower()
    medication_db['disease_norm'] = medication_db['disease'].astype(str).str.strip().str.lower()
    match = medication_db[medication_db['disease_norm'] == disease_norm]
    if match.empty:
        match = medication_db[medication_db['disease_norm'].str.contains(disease_norm)]
    if not match.empty:
        drug_classes = "; ".join(match['drug_classes'].dropna().unique())
        example_drugs = "; ".join(match['example_drugs'].dropna().unique())
        notes = "; ".join(match['key_notes'].dropna().unique())
        return {
            "drug_classes": drug_classes if drug_classes else "N/A",
            "example_drugs": example_drugs if example_drugs else "N/A",
            "key_notes": notes if notes else ""
        }
    else:
        return {
            "drug_classes": "OTC / Supportive Care",
            "example_drugs": "Paracetamol; Oral Rehydration; Rest",
            "key_notes": "General supportive measures for mild illness."
        }

def nearby_hospitals(lat, lon):
    return [
        ("City Hospital", lat + 0.01, lon + 0.01),
        ("Metro Clinic", lat - 0.01, lon - 0.02),
        ("Green Valley Hospital", lat + 0.015, lon - 0.015),
    ]

def load_logs(file):
    if os.path.exists(file):
        return pd.read_csv(file)
    else:
        return pd.DataFrame()

def save_log_to_file(log, file):
    df = load_logs(file)
    new_row = pd.DataFrame([log])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(file, index=False)

def check_health_alerts(row):
    problems = []
    general_fields = ["Fever", "Pain", "Difficulty Breathing", "Rested"]
    for field in general_fields:
        if field in row and ((field == "Rested" and row[field] == "No") or (field != "Rested" and row[field] == "Yes")):
            problems.append(field.lower().replace(" ", "_"))
    for key, val in row.items():
        if key not in general_fields + ["Date", "Diet", "Activity", "Symptoms", "Cough", "Fatigue", "Loss of Appetite", "Headache"]:
            if val == "No" and ("improv" in key.lower() or "less" in key.lower() or "reduce" in key.lower()):
                problems.append(key)
            if val == "Yes" and any(neg in key.lower() for neg in ["new", "any", "pain", "bleeding", "appeared"]):
                problems.append(key)
    return problems

def analyze_daily_progress(df):
    msgs = []
    if df.shape[0] < 2:
        return ["Not enough data to analyze daily progress."]
    recent = df.iloc[-1]
    previous = df.iloc[-2]
    key_symptoms = ["Fever","Pain","Difficulty Breathing","Fatigue","Loss of Appetite","Rested","Headache"]
    for symptom in key_symptoms:
        if symptom in recent and symptom in previous:
            rec = recent[symptom]
            prev = previous[symptom]
            if rec != prev:
                if rec == "No" and prev == "Yes":
                    msgs.append(f"Good news: your {symptom.lower()} symptom appears to have improved since your last log.")
                elif rec == "Yes" and prev == "No":
                    msgs.append(f"Alert: your {symptom.lower()} symptom has worsened compared to your previous log.")
    if not msgs:
        msgs.append("No significant changes detected in your recent health status compared to previous log.")
    return msgs

st.sidebar.header("Settings")
lang_choice = st.sidebar.selectbox("Language", ["en", "hi", "es", "fr"])
input_mode = st.sidebar.radio("Input mode", ["Type symptoms", "Select symptoms"])
show_map = st.sidebar.checkbox("Show nearby hospitals", True)
enable_doctor_conn = st.sidebar.checkbox("Enable doctor connection feature", False)
st.sidebar.caption("⚠️ For informational purposes only. Consult a healthcare professional.")

st.title("AI-Assisted Telemedicine Platform")

if input_mode == "Type symptoms":
    symptom_text = st.text_area("Describe your symptoms")
else:
    symptom_cols = [c for c in diseases_symptoms.columns if c != 'Disease']
    selected_symptoms = st.multiselect("Select symptoms", symptom_cols)
    symptom_text = ", ".join(selected_symptoms)

if st.button("Get Assessment"):
    if symptom_text.strip():
        symptom_english = symptom_text if lang_choice == "en" else translator.translate(symptom_text, dest='en').text
        predictions = predict_diseases(symptom_english, top_n=3)
        st.session_state['predictions'] = predictions
        st.session_state['med_choice'] = None
    else:
        st.warning("Please enter or select symptoms.")

if 'predictions' in st.session_state and st.session_state['predictions']:
    st.subheader("Preliminary Diagnosis")
    for disease, conf in st.session_state['predictions']:
        st.markdown(f"**{disease}** — Confidence: {conf:.1f}% — {triage_msg(conf)}")
    top_prediction = st.session_state['predictions'][0]
    primary_disease = top_prediction[0]
    primary_conf = top_prediction[1]
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Natural Remedies"):
            st.session_state['med_choice'] = "natural"
    with col2:
        if st.button("Medical Remedies"):
            st.session_state['med_choice'] = "medical"

    if st.session_state.get('med_choice'):
        if st.session_state['med_choice'] == "natural":
            st.markdown(f"#### Natural Remedies for {primary_disease}:")
            st.markdown("- Hydration\n- Rest\n- Balanced diet\n- Avoid stress")
        else:
            st.markdown(f"#### Medical Remedies for {primary_disease}")
            med_info = find_medications(primary_disease)
            st.write(f"**Drug Classes:** {med_info['drug_classes']}")
            st.write(f"**Example Drugs:** {med_info['example_drugs']}")
            st.write(f"**Notes:** {med_info['key_notes']}")
            st.caption("⚠️ For informational purposes only. Always consult a doctor.")

with st.expander("Disease Progress Tracker", expanded=True):
    chosen_diseases = st.multiselect("Select your disease(s)", list(DISEASE_PROGRESS_QUESTIONS.keys()))
    progress_answers = {}
    for dis in chosen_diseases:
        st.markdown(f'**{dis}**')
        qlist = DISEASE_PROGRESS_QUESTIONS[dis]
        for q in qlist:
            user_ans = st.selectbox(q, ["Yes", "No"], key=f"{dis}_{q}")
            progress_answers[f"{dis}: {q}"] = user_ans

    if st.button("Save Progress"):
        if chosen_diseases:
            progress_log = {"Date": str(datetime.date.today())}
            progress_log.update(progress_answers)
            save_log_to_file(progress_log, PROGRESS_FILE)
            st.success("Disease progress information saved successfully!")
        else:
            st.warning("Please select at least one disease before saving.")

with st.expander("Daily Log: Add or View"):
    with st.form("log_form"):
        col1, col2 = st.columns(2)
        with col1:
            log_date = st.date_input("Date", value=datetime.date.today())
            diet = st.text_area("Diet / Food intake", height=80)
            activity = st.text_area("Activity / Exercise", height=80)
            symptoms_update = st.text_area("Current symptoms / Changes", height=80)
        with col2:
            fever = st.selectbox("Fever", ["No", "Yes"])
            cough = st.selectbox("Cough", ["No", "Yes"])
            pain = st.selectbox("Pain", ["No", "Yes"])
            breathing = st.selectbox("Difficulty in Breathing", ["No", "Yes"])
            fatigue = st.selectbox("Fatigue / Tiredness", ["No", "Yes"])
            appetite = st.selectbox("Loss of Appetite", ["No", "Yes"])
            rested = st.selectbox("Well Rested?", ["Yes", "No"])
            headache = st.selectbox("Headache", ["No", "Yes"])
        submitted = st.form_submit_button("Save Log")
        if submitted:
            new_log = {
                "Date": str(log_date),
                "Diet": diet,
                "Activity": activity,
                "Symptoms": symptoms_update,
                "Fever": fever,
                "Cough": cough,
                "Pain": pain,
                "Difficulty Breathing": breathing,
                "Fatigue": fatigue,
                "Loss of Appetite": appetite,
                "Rested": rested,
                "Headache": headache
            }
            save_log_to_file(new_log, LOG_FILE)
            st.success("Daily log saved successfully!")

daily_logs_df = load_logs(LOG_FILE)
progress_logs_df = load_logs(PROGRESS_FILE)

if not daily_logs_df.empty or not progress_logs_df.empty:
    with st.expander("Progress History"):
        if not daily_logs_df.empty:
            st.subheader("Daily Logs")
            st.dataframe(daily_logs_df)
            msgs = analyze_daily_progress(daily_logs_df)
            for m in msgs:
                st.info(m)
        if not progress_logs_df.empty:
            st.subheader("Disease Progress Logs")
            st.dataframe(progress_logs_df)
        recent = {}
        if not daily_logs_df.empty:
            recent.update(daily_logs_df.iloc[-1].to_dict())
        if not progress_logs_df.empty:
            recent.update(progress_logs_df.iloc[-1].to_dict())
        alert_problems = check_health_alerts(recent)
        if alert_problems:
            st.warning(f"Health alert: {', '.join(alert_problems)} reported. Please consult a doctor urgently!")
        elif recent.get("Fatigue") == "Yes":
            st.info("You may be experiencing fatigue. Monitor your rest and recovery.")
        else:
            st.success("No major issues detected in your recent logs.")

if show_map:
    st.subheader("Nearby Hospitals")
    location_data = streamlit_geolocation()
    if location_data and location_data['latitude'] and location_data['longitude']:
        lat, lon = location_data['latitude'], location_data['longitude']
        st.success(f"Your Location: {lat:.5f}, {lon:.5f}")
        m = folium.Map(location=[lat, lon], zoom_start=14)
        folium.Marker([lat, lon], tooltip="You are here", icon=folium.Icon(color="blue", icon="user")).add_to(m)
        for place, pl_lat, pl_lon in nearby_hospitals(lat, lon):
            folium.Marker([pl_lat, pl_lon], tooltip=place, icon=folium.Icon(color="red", icon="plus")).add_to(m)
        st_folium(m, width=700, height=500)
    else:
        st.info("Please allow location access in your browser and reload.")

if enable_doctor_conn:
    st.subheader("Available Doctors for Teleconsultation")
    st.markdown("- Dr. Sharma (General Physician) - Available now")
    st.markdown("- Dr. Kumar (Internal Medicine) - Available in 30 mins")
    if st.button("Request Callback"):
        st.success("Request sent! A doctor will contact you shortly.")

if st.button("Clear All Logs and Results"):
    if os.path.exists(LOG_FILE): os.remove(LOG_FILE)
    if os.path.exists(PROGRESS_FILE): os.remove(PROGRESS_FILE)
    st.session_state.clear()
    st.experimental_rerun()
