import pandas as pd
import wikipediaapi
from datetime import datetime
import re
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# ------------------------------
# 📌 Function to Fetch DOB from Wikipedia
# ------------------------------
def get_dob_from_wikipedia(athlete_name):
    """
    Fetch the athlete's Date of Birth (DOB) from Wikipedia.
    """
    user_agent = "OlympicDataFetcher/1.0 (your-email@example.com)"  # Replace with your email
    wiki_wiki = wikipediaapi.Wikipedia(user_agent, "en")

    # Fetch Wikipedia page
    page = wiki_wiki.page(athlete_name)

    if not page.exists():
        return None

    # Use regex to search for DOB in page text
    dob_match = re.search(r'Born.*?(\d{1,2} \w+ \d{4})', page.text)

    if dob_match:
        dob_str = dob_match.group(1)
        try:
            # Convert DOB string to 'YYYY-MM-DD' format
            dob_date = datetime.strptime(dob_str, "%d %B %Y")
            return dob_date.strftime("%Y-%m-%d")
        except ValueError:
            return None

    return None

# ------------------------------
# 📌 Function to Calculate Age
# ------------------------------
def calculate_age(dob, olympic_year):
    """Calculate age from DOB and Olympic year."""
    dob_date = datetime.strptime(dob, "%Y-%m-%d")
    return olympic_year - dob_date.year

# ------------------------------
# 📌 Load Dataset & Filter Archery Medalists
# ------------------------------
df = pd.read_csv("athlete_events.csv")  # Replace with your dataset
df = df[(df["Sport"] == "Archery") & (df["Medal"].notna())]  # Filter Archery medalists

# Remove team events
df = df[~df["Event"].str.lower().str.contains("double|team", na=False)]

# ------------------------------
# 📌 Impute Missing Ages
# ------------------------------
df_missing_age = df[df["Age"].isna()]

for index, row in df_missing_age.iterrows():
    athlete_name = row["Name"]
    olympic_year = row["Year"]

    # Fetch DOB & Calculate Age
    dob = get_dob_from_wikipedia(athlete_name)
    if dob:
        age = calculate_age(dob, olympic_year)
        df.at[index, "Age"] = age  # Update Age

# Impute Remaining Missing Ages with Mean Age
df["Age"].fillna(df["Age"].mean(), inplace=True)

# ------------------------------
# 📌 Streamlit App
# ------------------------------
st.title("🏹 Archery Olympic Medalists - Age Distribution")

# Sidebar Filters
years = sorted(df["Year"].unique())
genders = df["Sex"].unique()
selected_year = st.sidebar.selectbox("Select Olympic Year", years, index=len(years)-1)
selected_gender = st.sidebar.radio("Select Gender", ["Both"] + list(genders))

# Apply Filters
filtered_df = df[df["Year"] == selected_year]
if selected_gender != "Both":
    filtered_df = filtered_df[filtered_df["Sex"] == selected_gender]

# 📊 Interactive Age Distribution Plot
fig = px.histogram(
    filtered_df, x="Age", color="Sex", barmode="overlay",
    title=f"Age Distribution of Archery Medalists ({selected_year})",
    labels={"Age": "Age", "count": "Number of Athletes"},
    color_discrete_map={"M": "blue", "F": "purple"}
)
st.plotly_chart(fig)

# 📌 Further Insights
st.subheader("📊 Age Insights")
mean_age = filtered_df["Age"].mean()
median_age = filtered_df["Age"].median()
age_range = (filtered_df["Age"].min(), filtered_df["Age"].max())

st.write(f"**Mean Age:** {mean_age:.2f} years")
st.write(f"**Median Age:** {median_age:.2f} years")
st.write(f"**Age Range:** {age_range[0]} - {age_range[1]} years")

# 📌 Interpretation
st.subheader("📖 Interpretation")
if mean_age < 25:
    st.write("Younger athletes tend to dominate Archery events, with a mean age below 25 years.")
elif mean_age < 30:
    st.write("The prime age for Archery medalists is typically between 25-30 years.")
else:
    st.write("Older athletes (30+ years) are also highly competitive in Archery.")

# 📌 Display Filtered Data
st.subheader("Filtered Archery Medalist Data")
st.dataframe(filtered_df)


################### Basic V1 ##########################

# import pandas as pd
# import wikipediaapi
# from datetime import datetime
# import re
# import streamlit as st
# import matplotlib.pyplot as plt
# import seaborn as sns

# # ------------------------------
# # 📌 Function to Fetch DOB from Wikipedia
# # ------------------------------
# def get_dob_from_wikipedia(athlete_name):
#     """
#     Fetch the athlete's Date of Birth (DOB) from Wikipedia.
#     """
#     user_agent = "OlympicDataFetcher/1.0 (your-email@example.com)"  # Replace with your email
#     wiki_wiki = wikipediaapi.Wikipedia(user_agent, "en")  # Corrected order

#     # Fetch Wikipedia page
#     page = wiki_wiki.page(athlete_name)

#     if not page.exists():
#         return None

#     # Use regex to search for DOB in page text
#     dob_match = re.search(r'Born.*?(\d{1,2} \w+ \d{4})', page.text)

#     if dob_match:
#         dob_str = dob_match.group(1)
#         try:
#             # Convert DOB string to 'YYYY-MM-DD' format
#             dob_date = datetime.strptime(dob_str, "%d %B %Y")
#             return dob_date.strftime("%Y-%m-%d")
#         except ValueError:
#             return None

#     return None

# # ------------------------------
# # 📌 Function to Calculate Age
# # ------------------------------
# def calculate_age(dob, olympic_year):
#     """Calculate age from DOB and Olympic year."""
#     dob_date = datetime.strptime(dob, "%Y-%m-%d")
#     return olympic_year - dob_date.year

# # ------------------------------
# # 📌 Load Dataset & Filter Archery Medalists
# # ------------------------------
# df = pd.read_csv("athlete_events.csv")  # Replace with your dataset
# df = df[(df["Sport"] == "Archery") & (df["Medal"].notna())]  # Filter Archery medalists

# # ------------------------------
# # 📌 Impute Missing Ages
# # ------------------------------
# df_missing_age = df[df["Age"].isna()]

# for index, row in df_missing_age.iterrows():
#     athlete_name = row["Name"]
#     olympic_year = row["Year"]

#     # Fetch DOB & Calculate Age
#     dob = get_dob_from_wikipedia(athlete_name)
#     if dob:
#         age = calculate_age(dob, olympic_year)
#         df.at[index, "Age"] = age  # Update Age

# # Impute Remaining Missing Ages with Mean Age
# df["Age"].fillna(df["Age"].mean(), inplace=True)

# # ------------------------------
# # 📌 Streamlit App
# # ------------------------------
# st.title("🏹 Archery Olympic Medalists - Age Distribution")

# # 📊 Plot Age Distribution by Gender
# fig, ax = plt.subplots(figsize=(8, 5))
# sns.histplot(data=df, x="Age", hue="Sex", kde=True, bins=15, ax=ax, palette={"M": "blue", "F": "purple"})
# plt.xlabel("Age")
# plt.ylabel("Count")
# plt.title("Age Distribution of Archery Medalists")
# st.pyplot(fig)

# # 📌 Display Filtered Data
# st.subheader("Filtered Archery Medalist Data")
# st.dataframe(df)



# import streamlit as st
# import requests
# import json
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import time

# def fetch_data(url):
#     try:
#         response = requests.get(url)
#         response.raise_for_status()
#         return response.json()
#     except requests.exceptions.RequestException as e:
#         st.error(f"An error occurred: {e}")
#         return None
#     except json.JSONDecodeError:
#         st.error("Failed to decode the response as JSON.")
#         return None

# def fetch_and_store_all_data(match_id):
#     base_url = "https://myfinal11.in/wp-admin/admin-ajax.php?action=wpec_api_request&path=matches%2F{MatchId}%2F{Path}"
#     # https://myfinal11.in/wp-admin/admin-ajax.php?action=wpec_api_request&path=matches%2F78132%2Flive
#     paths = {
#         "commentary_innings_1": "innings%2F1%2Fcommentary",
#         "commentary_innings_2": "innings%2F2%2Fcommentary",
#         "wagon": "wagons",
#         "statistics": "statistics",
#         "live": "live"
#     }

#     all_data = {}
#     for key, path in paths.items():
#         url = base_url.format(MatchId=match_id, Path=path)
#         data = fetch_data(url)
#         if data is not None:
#             all_data[key] = data
#             filename = f"responses/{key}.json"
#             os.makedirs("responses", exist_ok=True)
#             with open(filename, "w") as json_file:
#                 json.dump(data, json_file, indent=4)

#     return all_data

# def create_dataframes(all_data):
#     df_commentary_innings_1 = pd.DataFrame(all_data.get("commentary_innings_1", []))
#     df_commentary_innings_2 = pd.DataFrame(all_data.get("commentary_innings_2", []))
#     df_wagon = pd.DataFrame(all_data.get("wagon", []))
#     df_statistics = pd.DataFrame(all_data.get("statistics", {}).get("manhattan", []))

#     df_combined_commentary = pd.concat([df_commentary_innings_1, df_commentary_innings_2], ignore_index=True)

#     return df_combined_commentary, df_wagon, df_statistics

# def plot_manhattan_chart(df_statistics, placeholder):
#     with placeholder.container():
#         if "over" in df_statistics and "runs" in df_statistics:
#             plt.figure(figsize=(10, 5))
#             plt.bar(df_statistics['over'], df_statistics['runs'], color='blue')
#             plt.xlabel("Over")
#             plt.ylabel("Runs")
#             plt.title("Manhattan Chart")
#             st.pyplot(plt)
#         else:
#             st.warning("Required data for Manhattan Chart is missing.")

# def plot_worm_chart(df_statistics, placeholder):
#     with placeholder.container():
#         if "over" in df_statistics and "cumulative_runs" in df_statistics:
#             plt.figure(figsize=(10, 5))
#             plt.plot(df_statistics['over'], df_statistics['cumulative_runs'], marker='o')
#             plt.xlabel("Over")
#             plt.ylabel("Cumulative Runs")
#             plt.title("Worm Chart")
#             st.pyplot(plt)
#         else:
#             st.warning("Required data for Worm Chart is missing.")

# def plot_runrate_chart(df_statistics, placeholder):
#     with placeholder.container():
#         if "over" in df_statistics and "runrate" in df_statistics:
#             plt.figure(figsize=(10, 5))
#             plt.plot(df_statistics['over'], df_statistics['runrate'], marker='o', color='green')
#             plt.xlabel("Over")
#             plt.ylabel("Run Rate")
#             plt.title("Run Rate Chart")
#             st.pyplot(plt)
#         else:
#             st.warning("Required data for Run Rate Chart is missing.")

# def plot_extras_chart(df_statistics, placeholder):
#     with placeholder.container():
#         extras = df_statistics.get("extras", [])
#         if extras:
#             extras_df = pd.DataFrame(extras)
#             if "name" in extras_df and "value" in extras_df:
#                 plt.figure(figsize=(10, 5))
#                 plt.bar(extras_df['name'], extras_df['value'], color='orange')
#                 plt.xlabel("Extra Types")
#                 plt.ylabel("Count")
#                 plt.title("Extras Distribution")
#                 st.pyplot(plt)
#             else:
#                 st.warning("Extras data is incomplete.")
#         else:
#             st.warning("No data available for Extras Distribution.")

# def main():
#     st.title("Cricket Analyzer")

#     if 'all_data' not in st.session_state:
#         st.session_state.all_data = None
#         st.session_state.df_statistics = None

#     match_ids = {
#         "Match 1": "87106",
#         "Match 2": "87107",
#         "Match 3": "87108"
#     }

#     selected_match = st.selectbox("Select a Match", list(match_ids.keys()))
#     match_id = match_ids[selected_match]

#     if st.button("Fetch and Analyze Data"):
#         with st.spinner("Fetching data..."):
#             time.sleep(1)  # Simulate loading time
#             all_data = fetch_and_store_all_data(match_id)
#             df_commentary, df_wagon, df_statistics = create_dataframes(all_data)
#             st.session_state.all_data = all_data
#             st.session_state.df_statistics = df_statistics
#             st.success("Data fetched and processed successfully!")

#             st.subheader("Fetched Data")
#             st.write("### Statistics Data")
#             st.dataframe(df_statistics)
#             st.write("### Commentary Data")
#             st.dataframe(df_commentary)

#     if st.session_state.df_statistics is not None:
#         st.subheader("Visualizations")

#         chart_placeholder = st.empty()

#         if st.button("Manhattan Chart"):
#             with st.spinner("Loading Manhattan Chart..."):
#                 time.sleep(1)  # Simulate loading time
#                 plot_manhattan_chart(st.session_state.df_statistics, chart_placeholder)

#         if st.button("Worm Chart"):
#             with st.spinner("Loading Worm Chart..."):
#                 time.sleep(1)  # Simulate loading time
#                 plot_worm_chart(st.session_state.df_statistics, chart_placeholder)

#         if st.button("Run Rate Chart"):
#             with st.spinner("Loading Run Rate Chart..."):
#                 time.sleep(1)  # Simulate loading time
#                 plot_runrate_chart(st.session_state.df_statistics, chart_placeholder)

#         if st.button("Extras Distribution"):
#             with st.spinner("Loading Extras Distribution..."):
#                 time.sleep(1)  # Simulate loading time
#                 plot_extras_chart(st.session_state.df_statistics, chart_placeholder)

# if __name__ == "__main__":
#     main()
