import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Titanic Survival Predictor", layout="wide", page_icon="🚢")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    data = pd.read_csv("tested.csv")
    data['Age']      = data['Age'].fillna(data['Age'].median())
    data['Fare']     = data['Fare'].fillna(data['Fare'].median())
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

    train_df = data.copy()
    train_df['Sex']      = train_df['Sex'].map({'male': 0, 'female': 1})
    train_df['Embarked'] = train_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    return data, train_df

try:
    df, df_encoded = load_data()
except FileNotFoundError:
    st.error("Error: 'tested.csv' not found. Please upload it to your GitHub repository.")
    st.stop()

# --- MODEL TRAINING ---
@st.cache_resource
def train_models(data):
    X = data[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
    y = data['Survived']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    dt = DecisionTreeClassifier(random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    dt_acc = accuracy_score(y_test, dt.predict(X_test))
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    return dt, rf, dt_acc, rf_acc

dt_model, rf_model, dt_acc, rf_acc = train_models(df_encoded)

# --- SIDEBAR ---
st.sidebar.header("🚢 Passenger Details")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home & Predict", "📊 Data Visualizations", "📂 Bulk Scanner"]
)

pclass       = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
sex_raw      = st.sidebar.selectbox("Gender", ["male", "female"])
age          = st.sidebar.slider("Age", 1, 80, 25)
fare         = st.sidebar.slider("Fare Price", 0, 500, 50)
embarked_raw = st.sidebar.selectbox("Port of Embarkation", ["S", "C", "Q"])

sex      = 0 if sex_raw == "male" else 1
embarked = {"S": 0, "C": 1, "Q": 2}[embarked_raw]
input_data = np.array([[pclass, sex, age, fare, embarked]])

# ─── HELPER: preprocess uploaded CSV ─────────────────────────────
def preprocess_bulk(uploaded_df):
    df_proc = uploaded_df.copy()

    # Fill missing values — coerce to numeric first for safety
    if 'Age' in df_proc.columns:
        df_proc['Age'] = pd.to_numeric(df_proc['Age'], errors='coerce')
        df_proc['Age'] = df_proc['Age'].fillna(df_proc['Age'].median())
    if 'Fare' in df_proc.columns:
        df_proc['Fare'] = pd.to_numeric(df_proc['Fare'], errors='coerce')
        df_proc['Fare'] = df_proc['Fare'].fillna(df_proc['Fare'].median())
    if 'Embarked' in df_proc.columns:
        df_proc['Embarked'] = df_proc['Embarked'].fillna(
            df_proc['Embarked'].mode()[0]
        )

    # Encode Sex — use astype(str) so it always works regardless of pandas dtype
    if 'Sex' in df_proc.columns:
        sex_col = df_proc['Sex'].astype(str).str.strip().str.lower()
        df_proc['Sex'] = sex_col.map({'male': 0, 'female': 1})
        if df_proc['Sex'].isna().any():
            raise ValueError(
                "Sex column contains unrecognised values. "
                "Expected 'male' or 'female'."
            )

    # Encode Embarked — use astype(str) so it always works
    if 'Embarked' in df_proc.columns:
        emb_col = df_proc['Embarked'].astype(str).str.strip().str.upper()
        df_proc['Embarked'] = emb_col.map({'S': 0, 'C': 1, 'Q': 2})
        df_proc['Embarked'] = df_proc['Embarked'].fillna(0)  # fallback to S

    # Ensure Pclass is numeric
    if 'Pclass' in df_proc.columns:
        df_proc['Pclass'] = pd.to_numeric(df_proc['Pclass'], errors='coerce').fillna(3).astype(int)

    return df_proc

# ═══════════════════════════════════════════════════════════════════
# PAGE 1 — HOME & PREDICT
# ═══════════════════════════════════════════════════════════════════
if page == "🏠 Home & Predict":
    st.title("🚢 Titanic Survival Prediction App")
    st.markdown("Predict if a passenger would survive the Titanic disaster based on their details.")

    # KPI row
    total     = len(df)
    survived  = int(df['Survived'].sum())
    not_surv  = total - survived
    surv_rate = round(survived / total * 100, 1)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Passengers", f"{total}")
    k2.metric("Survived",         f"{survived}")
    k3.metric("Did Not Survive",  f"{not_surv}")
    k4.metric("Survival Rate",    f"{surv_rate}%")

    st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🤖 Make a Prediction")
        model_choice = st.selectbox("Choose AI Model", ["Decision Tree", "Random Forest"])

        acc = dt_acc if model_choice == "Decision Tree" else rf_acc
        st.caption(f"Model accuracy on test set: **{acc*100:.1f}%**")

        if st.button("Predict Survival Status", use_container_width=True):
            model    = dt_model if model_choice == "Decision Tree" else rf_model
            pred     = model.predict(input_data)[0]
            prob     = model.predict_proba(input_data)[0][1]

            if pred == 1:
                st.success(f"🎉 The passenger likely **Survived!**  (Confidence: {prob*100:.1f}%)")
            else:
                st.error(f"💀 The passenger likely **did not survive.**  (Confidence: {(1-prob)*100:.1f}%)")

            # Summary table
            st.markdown("**Input summary:**")
            summary = pd.DataFrame({
                "Feature": ["Class", "Gender", "Age", "Fare", "Embarked"],
                "Value":   [pclass, sex_raw, age, fare, embarked_raw]
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("📋 Dataset Preview")
        st.caption(f"Showing first 10 rows of tested.csv ({len(df)} total rows)")
        st.dataframe(df.head(10), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# PAGE 2 — DATA VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════
elif page == "📊 Data Visualizations":
    st.title("📊 Data Visualizations")
    st.markdown("Explore survival patterns from the Titanic dataset.")

    plot_option = st.selectbox(
        "Select Chart",
        [
            "Survival Count",
            "Gender Distribution",
            "Class Distribution",
            "Survival by Gender",
            "Survival by Class",
            "Age Distribution",
            "Fare Distribution",
        ]
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    if plot_option == "Survival Count":
        sns.countplot(x='Survived', data=df, ax=ax, palette="viridis")
        ax.set_xticklabels(["Did Not Survive (0)", "Survived (1)"])
        ax.set_title("Overall Survival Count")

    elif plot_option == "Gender Distribution":
        sns.countplot(x='Sex', data=df, ax=ax, palette="magma")
        ax.set_title("Gender Distribution")

    elif plot_option == "Class Distribution":
        sns.countplot(x='Pclass', data=df, ax=ax, palette="crest")
        ax.set_title("Passenger Class Distribution")
        ax.set_xlabel("Passenger Class")

    elif plot_option == "Survival by Gender":
        sns.countplot(x='Sex', hue='Survived', data=df, ax=ax, palette="Set2")
        ax.set_title("Survival by Gender")
        ax.legend(["Did Not Survive", "Survived"])

    elif plot_option == "Survival by Class":
        sns.countplot(x='Pclass', hue='Survived', data=df, ax=ax, palette="Set1")
        ax.set_title("Survival by Passenger Class")
        ax.set_xlabel("Passenger Class")
        ax.legend(["Did Not Survive", "Survived"])

    elif plot_option == "Age Distribution":
        ax.hist(df['Age'].dropna(), bins=30, color='steelblue', edgecolor='white')
        ax.set_title("Age Distribution")
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")

    elif plot_option == "Fare Distribution":
        ax.hist(df['Fare'].dropna(), bins=40, color='coral', edgecolor='white')
        ax.set_title("Fare Distribution")
        ax.set_xlabel("Fare (£)")
        ax.set_ylabel("Count")

    st.pyplot(fig)

# ═══════════════════════════════════════════════════════════════════
# PAGE 3 — BULK SCANNER
# ═══════════════════════════════════════════════════════════════════
elif page == "📂 Bulk Scanner":
    st.title("📂 Bulk Passenger Scanner")
    st.markdown(
        "Upload a CSV file with passenger details and get survival predictions for **all rows at once**."
    )

    # ── Required columns info ────────────────────────────────────
    with st.expander("ℹ️ Required CSV columns", expanded=False):
        st.markdown("""
Your CSV must contain these columns (same format as `tested.csv`):

| Column | Type | Description |
|---|---|---|
| `PassengerId` | int | Unique passenger ID |
| `Pclass` | int | Passenger class (1, 2, or 3) |
| `Name` | str | Full name |
| `Sex` | str | `male` or `female` |
| `Age` | float | Age in years (can have nulls) |
| `SibSp` | int | Siblings / spouses aboard |
| `Parch` | int | Parents / children aboard |
| `Ticket` | str | Ticket number |
| `Fare` | float | Ticket fare (can have nulls) |
| `Cabin` | str | Cabin number (can be empty) |
| `Embarked` | str | Port: `S`, `C`, or `Q` |

The `Survived` column is optional — if present, the scanner will show a comparison between actual and predicted values.
        """)

    # ── Model choice ─────────────────────────────────────────────
    bulk_model_choice = st.selectbox(
        "Choose model for bulk prediction",
        ["Random Forest", "Decision Tree"],
        key="bulk_model"
    )
    bulk_model = rf_model if bulk_model_choice == "Random Forest" else dt_model

    # ── Download sample CSV ───────────────────────────────────────
    st.markdown("#### 📥 Download Sample CSV")
    st.markdown(
        "Don't have a file? Download the built-in `tested.csv` dataset "
        "(418 passengers), then upload it below to try the scanner."
    )
    with open("tested.csv", "rb") as f:
        sample_csv_bytes = f.read()
    st.download_button(
        label="⬇️ Download tested.csv (sample dataset — 418 passengers)",
        data=sample_csv_bytes,
        file_name="tested.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.divider()

    # ── File uploader ─────────────────────────────────────────────
    st.markdown("#### 📤 Upload CSV for Bulk Prediction")
    uploaded_file = st.file_uploader(
        "Upload your CSV file", type=["csv"], key="bulk_upload"
    )

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded — **{len(raw_df)} passengers** found")

            # Validate required columns
            required = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
            missing_cols = [c for c in required if c not in raw_df.columns]
            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}")
                st.stop()

            # Preprocess & predict
            proc_df = preprocess_bulk(raw_df)
            X_bulk  = proc_df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]

            predictions  = bulk_model.predict(X_bulk)
            probabilities = bulk_model.predict_proba(X_bulk)[:, 1]

            # Build result dataframe
            result_df = raw_df.copy()
            result_df['Predicted_Survival']     = predictions
            result_df['Survival_Probability_%'] = (probabilities * 100).round(1)
            result_df['Prediction_Label']        = result_df['Predicted_Survival'].map(
                {1: "✅ Survived", 0: "❌ Did Not Survive"}
            )

            # ── KPI summary ──────────────────────────────────────
            st.divider()
            st.subheader("📊 Bulk Scan Summary")

            total_p   = len(result_df)
            pred_surv = int(predictions.sum())
            pred_not  = total_p - pred_surv
            avg_prob  = round(probabilities.mean() * 100, 1)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Passengers Scanned", f"{total_p}")
            m2.metric("Predicted Survived",        f"{pred_surv}",
                      f"{round(pred_surv/total_p*100,1)}%")
            m3.metric("Predicted Not Survived",    f"{pred_not}",
                      f"{round(pred_not/total_p*100,1)}%")
            m4.metric("Avg Survival Probability",  f"{avg_prob}%")

            # ── If actual Survived column exists — show accuracy ──
            if 'Survived' in raw_df.columns:
                actual = raw_df['Survived'].values
                match  = (predictions == actual).sum()
                acc    = round(match / total_p * 100, 1)
                st.info(
                    f"🎯 **Accuracy on this file:** {acc}%  "
                    f"({match} of {total_p} predictions matched actual Survived column)"
                )
                result_df['Actual_Survived'] = raw_df['Survived'].map(
                    {1: "✅ Survived", 0: "❌ Did Not Survive"}
                )
                result_df['Match'] = np.where(
                    predictions == actual, "✔ Correct", "✘ Wrong"
                )

            # ── Charts ───────────────────────────────────────────
            st.divider()
            st.subheader("📈 Visual Breakdown")

            ch1, ch2 = st.columns(2)

            with ch1:
                fig1, ax1 = plt.subplots(figsize=(5, 3.5))
                labels  = ["Did Not Survive", "Survived"]
                sizes   = [pred_not, pred_surv]
                colors  = ["#ef4444", "#22c55e"]
                ax1.pie(sizes, labels=labels, colors=colors,
                        autopct="%1.1f%%", startangle=90,
                        textprops={"fontsize": 11})
                ax1.set_title("Predicted Survival Split", fontsize=12)
                st.pyplot(fig1)

            with ch2:
                fig2, ax2 = plt.subplots(figsize=(5, 3.5))
                ax2.hist(probabilities * 100, bins=20,
                         color="steelblue", edgecolor="white")
                ax2.axvline(50, color="red", linestyle="--",
                            linewidth=1.5, label="50% threshold")
                ax2.set_title("Survival Probability Distribution", fontsize=12)
                ax2.set_xlabel("Survival Probability (%)")
                ax2.set_ylabel("Number of Passengers")
                ax2.legend()
                st.pyplot(fig2)

            # ── Class-wise breakdown chart ────────────────────────
            if 'Pclass' in result_df.columns:
                fig3, ax3 = plt.subplots(figsize=(7, 3.5))
                class_surv = result_df.groupby('Pclass')['Predicted_Survival'].mean() * 100
                ax3.bar(
                    [f"Class {c}" for c in class_surv.index],
                    class_surv.values,
                    color=["#2563EB", "#0891B2", "#D97706"]
                )
                ax3.set_title("Predicted Survival Rate by Passenger Class (%)", fontsize=12)
                ax3.set_ylabel("Survival Rate (%)")
                ax3.set_ylim(0, 110)
                for i, v in enumerate(class_surv.values):
                    ax3.text(i, v + 2, f"{v:.1f}%", ha='center', fontsize=11)
                st.pyplot(fig3)

            # ── Results table ─────────────────────────────────────
            st.divider()
            st.subheader("📋 Detailed Results Table")

            # Column order — show key columns first
            display_cols = ['PassengerId']
            if 'Name' in result_df.columns:
                display_cols.append('Name')
            display_cols += ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
            if 'Actual_Survived' in result_df.columns:
                display_cols.append('Actual_Survived')
            display_cols += ['Prediction_Label', 'Survival_Probability_%']
            if 'Match' in result_df.columns:
                display_cols.append('Match')

            # Filter only cols that exist
            display_cols = [c for c in display_cols if c in result_df.columns]

            # Search / filter
            filter_col1, filter_col2 = st.columns(2)
            with filter_col1:
                filter_pred = st.selectbox(
                    "Filter by prediction",
                    ["All", "✅ Survived", "❌ Did Not Survive"],
                    key="filter_pred"
                )
            with filter_col2:
                filter_class = st.multiselect(
                    "Filter by class",
                    options=[1, 2, 3],
                    default=[1, 2, 3],
                    key="filter_class"
                )

            display_df = result_df[display_cols].copy()
            if filter_pred != "All":
                display_df = display_df[
                    display_df['Prediction_Label'] == filter_pred
                ]
            if filter_class:
                display_df = display_df[
                    display_df['Pclass'].isin(filter_class)
                ]

            st.caption(f"Showing **{len(display_df)}** of **{total_p}** passengers")
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            # ── Download button ───────────────────────────────────
            st.divider()
            csv_out = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Full Results as CSV",
                data=csv_out,
                file_name="titanic_bulk_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

    else:
        # Show example / placeholder
        st.info(
            "👆 Upload a CSV file above to scan multiple passengers at once.  \n"
            "The file should have the same column structure as `tested.csv`."
        )
        st.markdown("**Example of expected input format:**")
        sample = pd.DataFrame({
            'PassengerId': [892, 893, 894],
            'Pclass':      [3, 3, 2],
            'Name':        ['Kelly, Mr. James', 'Wilkes, Mrs. James', 'Myles, Mr. Thomas'],
            'Sex':         ['male', 'female', 'male'],
            'Age':         [34.5, 47.0, 62.0],
            'SibSp':       [0, 1, 0],
            'Parch':       [0, 0, 0],
            'Ticket':      ['330911', '363272', '240276'],
            'Fare':        [7.83, 7.00, 9.69],
            'Cabin':       ['', '', ''],
            'Embarked':    ['Q', 'S', 'Q'],
        })
        st.dataframe(sample, use_container_width=True, hide_index=True)
