import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import json
from sklearn.exceptions import NotFittedError

# ---- App config ----
st.set_page_config(page_title="BigMart Ready Streamlit App", layout="wide", page_icon="📦")

# ---- Dynamic CSS (you can edit this block to change look & feel) ----
st.markdown(
    """
    <style>
    /* page background */
    .stApp {
        background: linear-gradient(120deg, #f6f9ff 0%, #ffffff 40%);
        font-family: 'Inter', sans-serif;
    }

    /* card-like container for inputs */
    .input-card {
        background: white;
        border-radius: 14px;
        box-shadow: 0 6px 18px rgba(50,50,93,0.06);
        padding: 18px;
    }

    /* headline */
    h1 {
        font-size: 28px;
        margin-bottom: 6px;
    }

    /* small helper text */
    .small {font-size:12px; color: #6b7280}

    /* buttons */
    div.stButton > button:first-child {
        border-radius: 10px;
        padding: 8px 18px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Helper functions ----

def load_model(path):
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        st.error(f"Could not load model from {path}: {e}")
        return None


def infer_features_from_model(model):
    # many sklearn models have attribute feature_names_in_
    if hasattr(model, 'feature_names_in_'):
        return list(model.feature_names_in_)
    # XGBoost sklearn wrapper sometimes stores _feature_names
    if hasattr(model, '_feature_names'):
        return list(model._feature_names)
    # tree boosters might store feature_names
    if hasattr(model, 'get_booster'):
        try:
            names = model.get_booster().feature_names
            if names:
                return list(names)
        except Exception:
            pass
    return None


def predict_df(model, df):
    try:
        preds = model.predict(df)
        return preds
    except NotFittedError:
        st.error("Model is not fitted.")
    except Exception as e:
        st.error(f"Prediction error: {e}")
    return None

# ---- Sidebar: Model and options ----
with st.sidebar:
    st.header("Model & options")
    st.write("Default model path: `/mnt/data/bigmart_best_model.pkl`")

    uploaded_model = st.file_uploader("Upload a .pkl joblib model (optional)", type=["pkl","joblib"], help="If you leave this empty the app will try to load the default model path.")
    model_path_input = st.text_input("Or provide model path", value="/mnt/data/bigmart_best_model.pkl")
    show_example = st.checkbox("Show example input dataset", value=True)

# --- Load model ---
model = None
if uploaded_model is not None:
    try:
        # streamlit file uploader returns a BytesIO; joblib can load from bytes via BytesIO
        model = joblib.load(uploaded_model)
        st.sidebar.success("Model loaded from upload")
    except Exception as e:
        st.sidebar.error(f"Failed loading uploaded model: {e}")
else:
    model = load_model(model_path_input)
    if model is not None:
        st.sidebar.success(f"Model loaded from {model_path_input}")

# ---- Main layout ----
st.title("BigMart — Ready-to-use Streamlit App")
st.write("Use this app to make single or batch predictions with your trained model. The app attempts to infer required features from the model; if it cannot, upload a CSV or provide JSON input.")

col1, col2 = st.columns([2,1])

with col1:
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    st.subheader("1) Provide input")
    st.write("You can: 1) upload a CSV for batch predictions, 2) fill a quick form for a single row, or 3) paste JSON representing a single row.")

    uploaded_csv = st.file_uploader("Upload CSV for batch predictions", type=["csv"], key="csv_uploader")

    # Try to infer feature names from model
    inferred_features = None
    if model is not None:
        inferred_features = infer_features_from_model(model)

    if inferred_features:
        st.info(f"Model wants these features: {inferred_features}")

    st.markdown("---")
    st.subheader("Single input (quick)")

    single_input_method = st.radio("How do you want to provide a single row?", ("Form (auto)", "JSON paste", "Manual - enter CSV-like row"))

    single_row_df = None
    if single_input_method == "Form (auto)":
        if inferred_features:
            with st.form("single_form"):
                values = {}
                cols = st.columns(2)
                for i, feat in enumerate(inferred_features):
                    col = cols[i % 2]
                    # fallback numeric input - user can later edit
                    values[feat] = col.text_input(feat, value="0")
                submitted = st.form_submit_button("Set single row")
                if submitted:
                    try:
                        # convert values to numeric where possible
                        parsed = {k: (float(v) if v.replace('.', '', 1).lstrip('-').isdigit() else v) for k, v in values.items()}
                        single_row_df = pd.DataFrame([parsed])
                        st.success("Single row prepared")
                    except Exception as e:
                        st.error(f"Could not construct row: {e}")
        else:
            st.warning("Model does not expose feature names. Use JSON or upload a CSV to set features.")

    elif single_input_method == "JSON paste":
        json_text = st.text_area("Paste single-row JSON (e.g. {\"Item_Weight\": 9.3, \"Item_Fat_Content\": \"Low Fat\"})")
        if st.button("Parse JSON"):
            try:
                parsed = json.loads(json_text)
                single_row_df = pd.DataFrame([parsed])
                st.success("JSON parsed into a single row")
            except Exception as e:
                st.error(f"JSON parse error: {e}")

    else:  # manual CSV-like
        row_text = st.text_input("Enter comma-separated values (order must match your model's feature order). Example: 9.3,Low Fat,0,92")
        if st.button("Parse row"):
            if inferred_features:
                vals = [v.strip() for v in row_text.split(",")]
                if len(vals) != len(inferred_features):
                    st.error(f"You provided {len(vals)} values but model expects {len(inferred_features)} features.")
                else:
                    single_row_df = pd.DataFrame([dict(zip(inferred_features, vals))])
                    st.success("Row created")
            else:
                st.error("Cannot parse — model feature names not known. Use JSON or upload CSV.")

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    st.subheader("2) Model & Prediction")
    if model is None:
        st.warning("No model loaded. Please upload one or set the correct path in the sidebar.")
    else:
        st.write("Model type:" , type(model))

    st.markdown("---")
    st.subheader("3) Make prediction")

    do_predict_single = st.button("Predict single row")
    do_predict_batch = st.button("Predict batch (CSV)")

    # storage for predictions
    if 'last_preds' not in st.session_state:
        st.session_state['last_preds'] = None
    if 'last_input_df' not in st.session_state:
        st.session_state['last_input_df'] = None

    if do_predict_single:
        if model is None:
            st.error("No model loaded")
        elif single_row_df is None:
            st.error("No single row prepared")
        else:
            # try to align columns
            df = single_row_df.copy()
            # try to convert dtypes
            for c in df.columns:
                try:
                    df[c] = pd.to_numeric(df[c])
                except Exception:
                    pass
            # if model expects features, reindex
            if inferred_features is not None:
                try:
                    df = df.reindex(columns=inferred_features)
                except Exception:
                    pass
            preds = predict_df(model, df)
            if preds is not None:
                st.success(f"Prediction: {preds[0] if len(preds)>0 else preds}")
                st.session_state['last_preds'] = preds
                st.session_state['last_input_df'] = df

    if do_predict_batch:
        if model is None:
            st.error("No model loaded")
        elif uploaded_csv is None:
            st.error("No CSV uploaded")
        else:
            try:
                df_batch = pd.read_csv(uploaded_csv)
                # align columns if we know them
                if inferred_features is not None:
                    missing = [f for f in inferred_features if f not in df_batch.columns]
                    if missing:
                        st.warning(f"Uploaded CSV is missing these features: {missing}")
                    df_batch = df_batch.reindex(columns=inferred_features)
                preds = predict_df(model, df_batch)
                if preds is not None:
                    df_out = df_batch.copy()
                    df_out['prediction'] = preds
                    st.session_state['last_preds'] = preds
                    st.session_state['last_input_df'] = df_out
                    st.success(f"Predicted {len(preds)} rows")
                    st.dataframe(df_out.head(20))
            except Exception as e:
                st.error(f"Failed batch predict: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# ---- Bottom area: download and examples ----
st.markdown("---")
col_a, col_b = st.columns([1,2])
with col_a:
    if st.session_state.get('last_input_df') is not None:
        to_download = st.session_state['last_input_df']
        csv_bytes = to_download.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime='text/csv')

with col_b:
    st.subheader("Example & tips")
    st.markdown("- If the model exposes `feature_names_in_` the app tries to auto-build a form.\n- For categorical columns use their encoded names the model expects (e.g., 'Low Fat' vs encoded 0/1).\n- If prediction fails, try uploading a CSV with the exact columns the model was trained on.")
    if show_example:
        st.write("Example input (first 6 rows) for guidance:")
        sample = pd.DataFrame({
            'Item_Identifier': ['FDA15','DRC01','FDP36','NCD19','FDO10','FDN15'],
            'Item_Weight': [9.3,5.92,17.5,19.2,8.93,10.395],
            'Item_Fat_Content': ['Low Fat','Regular','Low Fat','Low Fat','Regular','Low Fat'],
            'Item_Visibility': [0.016,0.019,0.016,0.000,0.045,0.013],
            'Item_Type': ['Dairy','Soft Drinks','Meat','Snack Foods','Household','Breads'],
            'Item_MRP': [249.8092,48.2692,141.6180,53.8614,182.0950,107.5520],
            'Outlet_Establishment_Year': [1999,2009,1998,1987,2002,1999],
            'Outlet_Size': ['Medium','Small','Medium','Small','High','Medium'],
            'Outlet_Location_Type': ['Tier 1','Tier 3','Tier 2','Tier 3','Tier 2','Tier 1'],
            'Outlet_Type': ['Supermarket Type1','Supermarket Type2','Grocery Store','Supermarket Type1','Supermarket Type1','Grocery Store']
        })
        st.dataframe(sample)

st.footer = st.markdown("<div style='text-align:center; padding:8px; color:#6b7280'>Built with ❤ — Edit the CSS at the top of the script to quickly change look & feel</div>", unsafe_allow_html=True)
