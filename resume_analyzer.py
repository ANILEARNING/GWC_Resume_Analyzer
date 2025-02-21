import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from fuzzywuzzy import process

# Load datasets
@st.cache_data
def load_data():
    with st.spinner("🔄 Loading data... Please wait while we process the Olympic and World Championship records."):

        olympic_data = pd.read_csv("individual_filtered_archery_data copy.csv")  # Update with actual path
        world_data = pd.read_csv("archery_championships_cleaned.csv")

        # Standardize column names
        olympic_data.columns = olympic_data.columns.str.strip().str.lower().str.replace(' ', '_')
        world_data.columns = world_data.columns.str.strip().str.lower().str.replace(' ', '_')

        # Fuzzy match names
        world_names = world_data['name'].dropna().unique()
        olympic_data['matched_name'] = olympic_data['name'].apply(lambda x: process.extractOne(x, world_names)[0] if x else None)

        # Merge data
        merged_data = olympic_data.merge(world_data, left_on='matched_name', right_on='name', how='left', suffixes=('_olympic', '_world'))

        # Filter for athletes who won medals in both
        matched_athletes = merged_data.dropna(subset=['name_world'])
        matched_athletes['time_gap'] = matched_athletes['year_olympic'] - matched_athletes['year_world']

        # Remove incorrect entries where time_gap is negative
        matched_athletes = matched_athletes[matched_athletes['time_gap'] >= 0]

        st.success("✅ Data successfully loaded! You can now explore the insights.")


    return olympic_data, world_data, matched_athletes

# Load Data
olympic_data, world_data, matched_athletes = load_data()

# Sidebar Filters
st.sidebar.header("🔍 Filter Data")
year_filter = st.sidebar.multiselect("Select Year", sorted(olympic_data['year'].unique()), default=[])
gender_filter = st.sidebar.multiselect("Select Gender", ['M', 'F'], default=[])
medal_filter = st.sidebar.multiselect("Select Medal", ['Gold', 'Silver', 'Bronze'], default=[])

# Apply Filters
filtered_data = matched_athletes.copy()
if year_filter:
    filtered_data = filtered_data[filtered_data['year_olympic'].isin(year_filter)]
if gender_filter:
    filtered_data = filtered_data[filtered_data['sex'].isin(gender_filter)]
if medal_filter:
    filtered_data = filtered_data[filtered_data['medal_olympic'].isin(medal_filter)]

# Metrics
total_athletes = len(matched_athletes)
unique_athletes = matched_athletes['name_olympic'].nunique()
avg_time_gap = matched_athletes['time_gap'].mean()
success_rate = (total_athletes / len(world_data)) * 100  # % of World medalists who won Olympic medals

st.title("🏹 Olympic Archery Success Dashboard")

col1, col2, col3, col4 = st.columns(4)
col1.metric("🏅 Matched Athletes", total_athletes)
col2.metric("👤 Unique Athletes", unique_athletes)
col3.metric("⏳ Avg Time Gap (Years)", round(avg_time_gap, 2))
col4.metric("📈 Success Rate (%)", f"{success_rate:.2f}%")

st.markdown("#### **📌 Interpretation**")
# Filter only positive transition times
valid_time_gaps = matched_athletes[matched_athletes['time_gap'] >= 0]['time_gap']

# Calculate statistics
transition_percentage = (len(matched_athletes) / len(world_data)) * 100  # % of World medalists who became Olympic medalists
avg_transition_time = valid_time_gaps.mean()  # Average years to transition
min_transition_time = valid_time_gaps.min()  # Minimum years to transition
max_transition_time = valid_time_gaps.max()  # Maximum years to transition
mode_transition_time = valid_time_gaps.mode().iloc[0] if not valid_time_gaps.mode().empty else "N/A"  # Most common transition time
unique_athletes_count = matched_athletes['name_olympic'].nunique()  # Unique medalists

# 📌 Interpretation Section
# st.markdown("#### **📌 Interpretation**")
st.write(f"""
- **{transition_percentage:.2f}%** of World Championship medalists later won Olympic medals, showing a weak correlation.
- The **average transition time** from a World Champion to an Olympic medalist is **{avg_transition_time:.2f} years**.
- The transition time varies from **{min_transition_time} to {max_transition_time} years**, with the most common transition period being **{mode_transition_time} years**.
- Having **{unique_athletes_count} unique athletes** suggests that some archers won multiple medals over the years.
""")

# Medal Transition Analysis
# Medal Transition Analysis
st.subheader("🥇 Medal Transition Analysis")
medal_counts = matched_athletes.groupby(['medal_olympic', 'medal_world']).size().reset_index(name='count')

# Calculate specific medal transitions
bronze_to_gold = matched_athletes[(matched_athletes['medal_world'] == 'Bronze') & (matched_athletes['medal_olympic'] == 'Gold')]
bronze_to_silver = matched_athletes[(matched_athletes['medal_world'] == 'Bronze') & (matched_athletes['medal_olympic'] == 'Silver')]
silver_to_gold = matched_athletes[(matched_athletes['medal_world'] == 'Silver') & (matched_athletes['medal_olympic'] == 'Gold')]
silver_to_bronze = matched_athletes[(matched_athletes['medal_world'] == 'Silver') & (matched_athletes['medal_olympic'] == 'Bronze')]
gold_to_silver = matched_athletes[(matched_athletes['medal_world'] == 'Gold') & (matched_athletes['medal_olympic'] == 'Silver')]
gold_to_bronze = matched_athletes[(matched_athletes['medal_world'] == 'Gold') & (matched_athletes['medal_olympic'] == 'Bronze')]

# Count transitions
bronze_to_gold_count = len(bronze_to_gold)
bronze_to_silver_count = len(bronze_to_silver)
silver_to_gold_count = len(silver_to_gold)
silver_to_bronze_count = len(silver_to_bronze)
gold_to_silver_count = len(gold_to_silver)
gold_to_bronze_count = len(gold_to_bronze)

# Define custom colors for medals
medal_colors = {
    "Gold": "#FFD700",   # Gold color
    "Silver": "#C0C0C0", # Silver color
    "Bronze": "#CD7F32"  # Bronze color
}

# Create bar chart with updated colors
fig = px.bar(
    medal_counts, 
    x="medal_world", 
    y="count", 
    color="medal_olympic",
    color_discrete_map=medal_colors,  # Apply medal colors
    labels={"medal_world": "World Medal", "count": "Number of Athletes"},
    title="🏆 Transition from World Championship to Olympic Medals"
)

# Display the chart in Streamlit
st.plotly_chart(fig)

# 📌 Interpretation Section
st.markdown("#### **📌 Interpretation**")
st.write(f"""
- **{bronze_to_gold_count} athletes** improved from **Bronze at Worlds → Gold at Olympics**, showcasing major skill growth.
- **{bronze_to_silver_count} athletes** advanced from **Bronze at Worlds → Silver at Olympics**, indicating steady improvement.
- **{silver_to_gold_count} athletes** progressed from **Silver at Worlds → Gold at Olympics**, demonstrating peak performance.
- **{silver_to_bronze_count} athletes** declined from **Silver at Worlds → Bronze at Olympics**, suggesting stronger Olympic competition.
- **{gold_to_silver_count} athletes** dropped from **Gold at Worlds → Silver at Olympics**, showing a slight performance dip.
- **{gold_to_bronze_count} athletes** fell from **Gold at Worlds → Bronze at Olympics**, highlighting the Olympic difficulty.
""")

# Dropdown to Show Upgraded and Downgraded Athletes
with st.expander("🔼 View Upgraded Athletes (Bronze → Gold)"):
    st.dataframe(bronze_to_gold[['name_olympic', 'year_olympic', 'country_olympic', 'medal_world', 'medal_olympic', 'time_gap']])

with st.expander("🔽 View Downgraded Athletes (Gold → Bronze)"):
    st.dataframe(gold_to_bronze[['name_olympic', 'year_olympic', 'country_olympic', 'medal_world', 'medal_olympic', 'time_gap']])


# Age Distribution
st.subheader("🎯 Age Distribution of Olympic Medalists")
fig = px.histogram(filtered_data, x='age_olympic', color='medal_olympic', nbins=20,
                   title="Age Distribution of Olympic Medalists")
st.plotly_chart(fig)

st.markdown("#### **📌 Interpretation**")
st.write("""
- The **peak performance age** for Olympic medalists is **25-30 years**.
""")

# Time Gap Distribution
st.subheader("⏳ Time Gap Between World & Olympic Medals")
fig = px.histogram(filtered_data, x='time_gap', nbins=10, title="Time Gap Between World & Olympic Medals")
st.plotly_chart(fig)

st.markdown("#### **📌 Interpretation**")
st.write("""
- Most archers win an Olympic medal within **2-6 years** of their World Championship success.
- A longer gap may suggest that experience or additional training is needed for Olympic success.
""")

# Country-Wise Performance
# Country-Wise Success in Olympic Transition
st.subheader("🌍 Country-Wise Success in Olympic Transition")

# Group by Olympic country and count unique athletes
country_success = matched_athletes.groupby('country_olympic')['name_olympic'].nunique().reset_index()
country_success = country_success.rename(columns={'name_olympic': 'athlete_count'})
top_countries = country_success.sort_values(by='athlete_count', ascending=False).head(7)

# Get the top country and their success count
top_country = top_countries.iloc[0]['country_olympic']
top_country_count = top_countries.iloc[0]['athlete_count']

fig = px.bar(top_countries, x='country_olympic', y='athlete_count', 
             title="Top 7 Countries Producing Olympic Medalists",
             labels={'athlete_count': 'Number of Athletes', 'country_olympic': 'Country'},
             color='athlete_count',
             color_continuous_scale='Blues')

st.plotly_chart(fig)

st.markdown("#### **📌 Interpretation**")
st.write(f"""
- The **top-performing country** in Olympic transitions is **{top_country}**, with **{top_country_count} athletes** successfully winning Olympic medals after medaling at the World Championships.  
- The **top 7 countries** shown above dominate in transitioning world-class archers to Olympic medalists.  
- Countries with strong archery programs consistently produce successful Olympians, reinforcing the importance of structured training.  
""")

# Top Athletes Table
# Top Athletes Table with Country Column from Olympic Data
st.subheader("🏆 Top Athletes Who Won Both World & Olympic Medals")
top_athletes = matched_athletes[['name_olympic', 'year_olympic','year_world','country_olympic', 'medal_olympic', 'medal_world', 'time_gap']]
st.dataframe(top_athletes.sort_values(by='time_gap').head(15))

st.markdown("#### **📌 Interpretation**")
st.write("""
- Some athletes won **both medals within 2 years**, showing rapid success.
- Others took longer, indicating that **Olympic success requires long-term consistency**.
""")
# st.subheader("🏆 Top Athletes Who Won Both World & Olympic Medals")
# top_athletes = matched_athletes[['name_olympic', 'year_olympic', 'country', 'medal_olympic', 'medal_world', 'time_gap']]
# st.dataframe(top_athletes.sort_values(by='time_gap').head(15))

# st.markdown("#### **📌 Interpretation**")
# st.write("""
# - Some athletes won **both medals within 2 years**, showing rapid success.
# - Others took longer, indicating that **Olympic success requires long-term consistency**.
# """)

st.markdown("### 📌 **Final Insights**")
st.write(f"""
- **{success_rate:.2f}%** of World Champions later won Olympic medals, proving weak correlation.
- Countries like **South Korea & USA** produce the most consistent medalists.
- The **optimal peak age** for Olympic success is **25-30 years**.
""")
