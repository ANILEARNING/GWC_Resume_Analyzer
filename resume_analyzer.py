import pandas as pd
import wikipediaapi
from datetime import datetime
import re
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(layout="wide", page_title="Olympic Archery Analytics")

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stSelectbox, .stRadio {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    h1 {
        color: #1f77b4;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Functions remain the same as in your original code
def get_dob_from_wikipedia(athlete_name):
    """Fetch the athlete's Date of Birth (DOB) from Wikipedia."""
    user_agent = "OlympicDataFetcher/1.0 (your-email@example.com)"
    wiki_wiki = wikipediaapi.Wikipedia(user_agent, "en")
    page = wiki_wiki.page(athlete_name)
    
    if not page.exists():
        return None
    
    dob_match = re.search(r'Born.*?(\d{1,2} \w+ \d{4})', page.text)
    if dob_match:
        dob_str = dob_match.group(1)
        try:
            dob_date = datetime.strptime(dob_str, "%d %B %Y")
            return dob_date.strftime("%Y-%m-%d")
        except ValueError:
            return None
    return None

def calculate_age(dob, olympic_year):
    """Calculate age from DOB and Olympic year."""
    dob_date = datetime.strptime(dob, "%Y-%m-%d")
    return olympic_year - dob_date.year

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv("athlete_events.csv")
    df = df[(df["Sport"] == "Archery") & (df["Medal"].notna())]
    
    # Create Team/Individual column
    df['Event_Type'] = df['Event'].apply(
        lambda x: 'Team' if any(word in x.lower() for word in ['team', 'double']) else 'Individual'
    )
    
    # Handle missing ages (your existing code for age imputation)
    df_missing_age = df[df["Age"].isna()]
    for index, row in df_missing_age.iterrows():
        dob = get_dob_from_wikipedia(row["Name"])
        if dob:
            age = calculate_age(dob, row["Year"])
            df.at[index, "Age"] = age
    
    # df["Age"].fillna(df["Age"].mean(), inplace=True)
    df.loc[df["Age"].isna() & (df["Sex"] == "M"), "Age"] = round(df[df["Sex"] == "M"]["Age"].mean())
    df.loc[df["Age"].isna() & (df["Sex"] == "F"), "Age"] = round(df[df["Sex"] == "F"]["Age"].mean())

    return df

# Load data
df = load_data()

# Main title
st.title("🏹 Olympic Archery Analytics Dashboard")
st.markdown("---")

# Filters in main content area
col1, col2, col3, col4 = st.columns(4)

with col1:
    years = sorted(df["Year"].unique())
    selected_year = st.selectbox(
        "Select Olympic Year",
        ["All"] + list(years),
        index=0
    )

with col2:
    selected_gender = st.radio(
        "Select Gender",
        ["Both", "M", "F"],
        horizontal=True
    )

with col3:
    event_types = df["Event_Type"].unique()
    selected_event = st.selectbox(
        "Select Event Type",
        ["All"] + list(event_types)
    )

with col4:
    seasons = df["Season"].unique()
    selected_season = st.selectbox(
        "Select Season",
        ["All"] + list(seasons)
    )

# Filter data based on selections
filtered_df = df.copy()
if selected_year != "All":
    filtered_df = filtered_df[filtered_df["Year"] == selected_year]
if selected_gender != "Both":
    filtered_df = filtered_df[filtered_df["Sex"] == selected_gender]
if selected_event != "All":
    filtered_df = filtered_df[filtered_df["Event_Type"] == selected_event]
if selected_season != "All":
    filtered_df = filtered_df[filtered_df["Season"] == selected_season]

# Key Metrics
st.markdown("### 📊 Key Metrics")
metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)

with metric_col1:
    st.metric("Unique Athletes", len(filtered_df["Name"].unique()))
with metric_col2:
    st.metric("Medalist Count", len(filtered_df["Name"]))
with metric_col3:
    gender_ratio = filtered_df["Sex"].value_counts(normalize=True)
    female_pct = gender_ratio.get("F", 0) * 100
    st.metric("Female Athletes", f"{female_pct:.1f}%")
with metric_col4:
    st.metric("Average Age", f"{filtered_df['Age'].mean():.1f}")
with metric_col5:
    st.metric("Age Range", f"{filtered_df['Age'].min():.0f} - {filtered_df['Age'].max():.0f}")

# Age Distribution Chart (similar to the image)
st.markdown("### 🎯 Age Distribution by Gender")

# Create mirrored histogram
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, 
                    row_heights=[0.5, 0.5])

# Female distribution (top)
female_df = filtered_df[filtered_df["Sex"] == "F"]
fig.add_trace(
    go.Bar(x=female_df["Age"].value_counts().sort_index().index,
           y=female_df["Age"].value_counts().sort_index().values,
           name="Female",
           marker_color="#FF69B4"),
    row=1, col=1
)

# Male distribution (bottom, negative values)
male_df = filtered_df[filtered_df["Sex"] == "M"]
fig.add_trace(
    go.Bar(x=male_df["Age"].value_counts().sort_index().index,
           y=-male_df["Age"].value_counts().sort_index().values,
           name="Male",
           marker_color="#4169E1"),
    row=2, col=1
)

fig.update_layout(
    height=600,
    showlegend=True,
    title_text="Age Distribution by Gender",
    xaxis2_title="Age",
    yaxis_title="Female Count",
    yaxis2_title="Male Count"
)

st.plotly_chart(fig, use_container_width=True)

# Additional visualizations
# col1, col2 = st.columns(2)

# with col1:
st.markdown("### 📈 Age Trends Over Time")
age_trends = px.box(filtered_df, x="Year", y="Age", color="Sex",
                    title="Age Distribution Across Olympics")
st.plotly_chart(age_trends, use_container_width=True)

# with col2:
    # st.markdown("### 🏅 Medal Distribution by Age Group")
    # filtered_df["Age_Group"] = pd.cut(filtered_df["Age"], 
    #                                 bins=[0, 20, 25, 30, 35, 100],
    #                                 labels=["Under 20", "20-25", "26-30", "31-35", "Over 35"])
    
    # medal_dist = px.bar(filtered_df, x="Age_Group", color="Medal",
    #                    title="Medals by Age Group",
    #                    category_orders={"Medal": ["Gold", "Silver", "Bronze"]})
    # st.plotly_chart(medal_dist, use_container_width=True)
st.markdown("### 🏅 Medal Distribution by Age Group")
    
filtered_df["Age_Group"] = pd.cut(filtered_df["Age"], 
                                    bins=[0, 20, 25, 30, 35, 100],
                                    labels=["Under 20", "20-25", "26-30", "31-35", "Over 35"])

# Group by Age_Group and Medal, then count medals
medal_counts = filtered_df.groupby(["Age_Group", "Medal"]).size().reset_index(name="count")

# Calculate total medals per Age_Group
total_medals = medal_counts.groupby("Age_Group")["count"].sum().reset_index()
total_medals_dict = dict(zip(total_medals["Age_Group"], total_medals["count"]))  # Dictionary for annotation

# Create bar chart
medal_dist = px.bar(
    medal_counts, x="Age_Group", y="count", color="Medal",
    title="Medals by Age Group",
    category_orders={"Medal": ["Gold", "Silver", "Bronze"]},  # Medal order
    text=medal_counts["count"],  # Show medal count inside bars
    color_discrete_map={"Gold": "#FFD700", "Silver": "#C0C0C0", "Bronze": "#CD7F32"}  # Medal colors
)

# Add total medal count on top of each bar
for age_group, total in total_medals_dict.items():
    medal_dist.add_annotation(
        x=age_group, y=total + 1,  # Position above bar
        text=str(total), showarrow=False,
        font=dict(size=14, color="black")
    )

st.plotly_chart(medal_dist, use_container_width=True)

# Enhanced Interpretation
st.markdown("### 📋 Statistical Insights")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Age Distribution Analysis")
    mean_age = filtered_df["Age"].mean()
    median_age = filtered_df["Age"].median()
    mode_age = filtered_df["Age"].mode().iloc[0]
    
    st.write(f"""
    - **Mean Age:** {mean_age:.1f} years
    - **Median Age:** {median_age:.1f} years
    - **Most Common Age:** {mode_age:.0f} years
    """)
    
    if mean_age < 25:
        st.write("The data suggests a younger athlete population dominates the sport.")
    elif mean_age < 30:
        st.write("Athletes in their prime (25-30) show strong representation.")
    else:
        st.write("Experience appears to be a significant factor with older athletes showing strong presence.")

with col2:
    st.markdown("#### Gender Analysis")
    gender_counts = filtered_df["Sex"].value_counts()
    st.write(f"""
    - **Female Athletes:** {gender_counts.get('F', 0)} ({(gender_counts.get('F', 0) / len(filtered_df) * 100):.1f}%)
    - **Male Athletes:** {gender_counts.get('M', 0)} ({(gender_counts.get('M', 0) / len(filtered_df) * 100):.1f}%)
    """)
    
    # Gender-specific age insights
    female_mean = filtered_df[filtered_df["Sex"] == "F"]["Age"].mean()
    male_mean = filtered_df[filtered_df["Sex"] == "M"]["Age"].mean()
    st.write(f"""
    - **Average Female Age:** {female_mean:.1f} years
    - **Average Male Age:** {male_mean:.1f} years
    """)

# Display filtered data
st.markdown("### 📋 Detailed Data View")
st.dataframe(filtered_df[["Name", "Sex", "Age", "Year", "NOC","Event", "Medal", "Event_Type", "Season"]])


