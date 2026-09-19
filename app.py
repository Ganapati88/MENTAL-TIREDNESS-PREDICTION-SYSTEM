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

# 2. Main Page Header
st.title('🧠 Mental Tiredness Score Predictor')
st.write(
    'Enter your daily metrics and profile details below to predict your mental'
    ' tiredness score instantly.'
)
st.markdown('---')

# 3. Daily Metrics & Habits at the START (starting at min_value so they aren't blank)
st.subheader('📊 Daily Metrics & Habits')
col1, col2, col3 = st.columns(3)

with col1:
  st.markdown('**Work & Tasks**')
  workload_score = st.number_input(
      'Workload Score (1-10)', min_value=1.0, max_value=10.0, value=1.0, step=0.1
  )
  task_complexity_avg = st.number_input(
      'Task Complexity Avg (1-10)',
      min_value=1.0,
      max_value=10.0,
      value=1.0,
      step=0.1,
  )
  number_of_decisions_made = st.number_input(
      'Number of Decisions Made', min_value=82, max_value=165, value=82, step=1
  )
  context_switch_count = st.number_input(
      'Context Switch Count', min_value=0, max_value=21, value=0, step=1
  )
  break_frequency = st.number_input(
      'Break Frequency', min_value=0, max_value=13, value=0, step=1
  )

with col2:
  st.markdown('**Digital & Focus**')
  screen_time_min = st.number_input(
      'Screen Time (minutes)',
      min_value=20.0,
      max_value=852.2,
      value=20.0,
      step=1.0,
  )
  deep_work_min = st.number_input(
      'Deep Work (minutes)',
      min_value=0.0,
      max_value=312.8,
      value=0.0,
      step=1.0,
  )
  notifications_received = st.number_input(
      'Notifications Received', min_value=30, max_value=101, value=30, step=1
  )
  caffeine_mg = st.number_input(
      'Caffeine Intake (mg)', min_value=0.0, max_value=422.9, value=0.0, step=1.0
  )

with col3:
  st.markdown('**Health & Environment**')
  sleep_hours = st.number_input(
      'Sleep Hours', min_value=3.0, max_value=11.0, value=3.0, step=0.1
  )
  deep_sleep_pct = st.number_input(
      'Deep Sleep %', min_value=0.0, max_value=41.38, value=0.0, step=0.1
  )
  hydration_l = st.number_input(
      'Hydration (Liters)', min_value=0.3, max_value=4.57, value=0.3, step=0.1
  )
  noise_level_db = st.number_input(
      'Noise Level (dB)', min_value=20.0, max_value=84.8, value=20.0, step=0.1
  )
  temperature_c = st.number_input(
      'Temperature (°C)', min_value=15.0, max_value=35.0, value=15.0, step=0.1
  )

st.markdown('---')

# 4. Profile & Environment at the BOTTOM with unselected placeholders
st.subheader('👤 Profile & Environment')
col_cat1, col_cat2, col_cat3 = st.columns(3)

with col_cat1:
  mood = st.selectbox(
      'Current Mood',
      ['-- Select Mood --', 'Happy', 'Neutral', 'Low'],
      index=0,
  )
with col_cat2:
  work_type = st.selectbox(
      'Work Type',
      ['-- Select Work Type --', 'Remote', 'Office', 'Manual', 'Student'],
      index=0,
  )
with col_cat3:
  work_environment = st.selectbox(
      'Work Environment',
      [
          '-- Select Work Environment --',
          'Quiet',
          'Moderate Noise',
          'Noisy',
      ],
      index=0,
  )

st.markdown('---')

# 5. Predict Button and Validation Logic
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
predict_btn = st.button(
    '🚀 Predict Mental Tiredness Score', use_container_width=True
)
st.markdown('</div>', unsafe_allow_html=True)

if predict_btn:
  # Validation check for unselected dropdowns
  if (
      mood.startswith('--')
      or work_type.startswith('--')
      or work_environment.startswith('--')
  ):
    st.warning(
        '⚠️ Please select valid options for Mood, Work Type, and Work'
        ' Environment before predicting!'
    )
  else:
    # Compile inputs into a DataFrame matching training schema
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

    prediction = pipeline.predict(input_data)[0]

    st.markdown('### Results')
    st.success(f'### Predicted Mental Tiredness Score: {prediction:.2f} / 100')

    if prediction < 20:
      st.info('🟢 Low fatigue level. Great balance today!')
    elif prediction < 45:
      st.warning(
          '🟡 Moderate fatigue. Consider taking a short break or hydrating.'
      )
    else:
      st.error(
          '🔴 High mental tiredness detected! Prioritize rest and logging off.'
      )