import pickle
import pandas as pd
import sklearn.compose._column_transformer as _ct
import streamlit as st

# --- COMPATIBILITY PATCH FOR PICKLE/SKLEARN VERSION DRIFT ---
if not hasattr(_ct, '_RemainderColsList'):

  class _RemainderColsList:
    pass

  _ct._RemainderColsList = _RemainderColsList
# -------------------------------------------------------------


# 1. Load the single pipeline file using pickle
@st.cache_resource
def load_pipeline():
  with open('mental_tiredness_model.pkl', 'rb') as f:
    pipeline = pickle.load(f)
  return pipeline




pipeline = load_pipeline()

# 2. Streamlit UI Layout
st.title('🧠 Mental Tiredness Score Predictor')
st.write(
    'Enter your daily work habits, environment, and metrics below to predict'
    ' your mental tiredness score instantly.'
)

# Sidebar inputs for numerical features
st.sidebar.header('Input Parameters')

number_of_decisions_made = st.sidebar.slider(
    'Number of Decisions Made', 82, 165, 120
)
context_switch_count = st.sidebar.slider('Context Switch Count', 0, 21, 8)
notifications_received = st.sidebar.slider('Notifications Received', 30, 101, 65)
screen_time_min = st.sidebar.slider(
    'Screen Time (minutes)', 20.0, 852.2, 302.0, step=5.0
)
deep_work_min = st.sidebar.slider(
    'Deep Work (minutes)', 0.0, 312.8, 94.8, step=5.0
)
task_complexity_avg = st.sidebar.slider(
    'Task Complexity Avg (1-10)', 1.0, 10.0, 5.5, step=0.1
)
caffeine_mg = st.sidebar.slider('Caffeine Intake (mg)', 0.0, 422.9, 129.0)
break_frequency = st.sidebar.slider('Break Frequency', 0, 13, 4)
sleep_hours = st.sidebar.slider('Sleep Hours', 3.0, 11.0, 6.9, step=0.1)
deep_sleep_pct = st.sidebar.slider('Deep Sleep %', 0.0, 41.38, 18.98, step=0.1)
hydration_l = st.sidebar.slider('Hydration (Liters)', 0.3, 4.57, 1.89, step=0.1)
noise_level_db = st.sidebar.slider('Noise Level (dB)', 20.0, 84.8, 47.9)
temperature_c = st.sidebar.slider('Temperature (°C)', 15.0, 35.0, 23.0)
workload_score = st.sidebar.slider('Workload Score (1-10)', 1.0, 10.0, 5.7, step=0.1)

# Categorical features dropdowns
mood = st.sidebar.selectbox('Current Mood', ['Happy', 'Neutral', 'Low'])
work_type = st.sidebar.selectbox(
    'Work Type', ['Remote', 'Office', 'Manual', 'Student']
)
work_environment = st.sidebar.selectbox(
    'Work Environment', ['Quiet', 'Moderate Noise', 'Noisy']
)

# 3. Compile raw inputs into a DataFrame matching training schema
input_data = pd.DataFrame({
    'number_of_decisions_made': [number_of_decisions_made],
    'context_switch_count': [context_switch_count],
    'notifications_received': [notifications_received],
    'screen_time_min': [screen_time_min],
    'deep_work_min': [deep_work_min],
    'task_complexity_avg': [task_complexity_avg],
    'caffeine_mg': [caffeine_mg],
    'break_frequency': [break_frequency],
    'sleep_hours': [sleep_hours],
    'deep_sleep_pct': [deep_sleep_pct],
    'hydration_l': [hydration_l],
    'mood': [mood],
    'work_type': [work_type],
    'work_environment': [work_environment],
    'noise_level_db': [noise_level_db],
    'temperature_c': [temperature_c],
    'workload_score': [workload_score],
})

# 4. Predict button
if st.button('Predict Mental Tiredness'):
  # Pipeline automatically handles encoding, scaling, selection, and prediction!
  prediction = pipeline.predict(input_data)[0]

  # Display result
  st.success(f'### Predicted Mental Tiredness Score: {prediction:.2f} / 100')

  if prediction < 20:
    st.info('🟢 Low fatigue level. Great balance today!')
  elif prediction < 45:
    st.warning('🟡 Moderate fatigue. Consider taking a short break or hydrating.')
  else:
    st.error(
        '🔴 High mental tiredness detected! Prioritize rest and logging off.'
    )