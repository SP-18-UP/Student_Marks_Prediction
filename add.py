import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Marks Predictor", page_icon="🎓")


@st.cache_resource
def load_model():
    return joblib.load("student_marks_model.pkl")


model = load_model()

# Range of the data the model was trained on
HOURS_RANGE = (1, 10)
ATTENDANCE_RANGE = (50, 90)

st.title("🎓 Student Marks Predictor")
st.caption("Multiple Linear Regression: Study Hours + Attendance → Expected Marks")

study_hours = st.slider("Study hours per day", 0.0, 12.0, 5.0, step=0.5)
attendance = st.slider("Attendance (%)", 0, 100, 75)

if st.button("Predict Marks", type="primary"):
    # Column names must match the ones used during training
    new_student = pd.DataFrame(
        {"Study_Hours": [study_hours], "Attendance": [attendance]}
    )

    prediction = float(model.predict(new_student)[0])
    prediction = max(0.0, min(100.0, prediction))  # keep marks between 0 and 100

    st.metric("Predicted Marks", f"{prediction:.2f} / 100")
    st.progress(int(prediction))

    out_of_range = not (
        HOURS_RANGE[0] <= study_hours <= HOURS_RANGE[1]
        and ATTENDANCE_RANGE[0] <= attendance <= ATTENDANCE_RANGE[1]
    )
    if out_of_range:
        st.warning(
            f"Your input is outside the training data "
            f"(study hours {HOURS_RANGE[0]}-{HOURS_RANGE[1]}, "
            f"attendance {ATTENDANCE_RANGE[0]}-{ATTENDANCE_RANGE[1]}%). "
            "The prediction may be unreliable."
        )

with st.expander("About this model"):
    st.write(f"Intercept: `{model.intercept_:.2f}`")
    st.write(
        {
            "Study_Hours": round(float(model.coef_[0]), 3),
            "Attendance": round(float(model.coef_[1]), 3),
        }
    )
