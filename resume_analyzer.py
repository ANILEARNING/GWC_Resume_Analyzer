import streamlit as st
import requests
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import time

def fetch_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred: {e}")
        return None
    except json.JSONDecodeError:
        st.error("Failed to decode the response as JSON.")
        return None

def fetch_and_store_all_data(match_id):
    base_url = "https://myfinal11.in/wp-admin/admin-ajax.php?action=wpec_api_request&path=matches%2F{MatchId}%2F{Path}"
    # https://myfinal11.in/wp-admin/admin-ajax.php?action=wpec_api_request&path=matches%2F78132%2Flive
    paths = {
        "commentary_innings_1": "innings%2F1%2Fcommentary",
        "commentary_innings_2": "innings%2F2%2Fcommentary",
        "wagon": "wagons",
        "statistics": "statistics",
        "live": "live"
    }

    all_data = {}
    for key, path in paths.items():
        url = base_url.format(MatchId=match_id, Path=path)
        data = fetch_data(url)
        if data is not None:
            all_data[key] = data
            filename = f"responses/{key}.json"
            os.makedirs("responses", exist_ok=True)
            with open(filename, "w") as json_file:
                json.dump(data, json_file, indent=4)

    return all_data

def create_dataframes(all_data):
    df_commentary_innings_1 = pd.DataFrame(all_data.get("commentary_innings_1", []))
    df_commentary_innings_2 = pd.DataFrame(all_data.get("commentary_innings_2", []))
    df_wagon = pd.DataFrame(all_data.get("wagon", []))
    df_statistics = pd.DataFrame(all_data.get("statistics", []))

    df_combined_commentary = pd.concat([df_commentary_innings_1, df_commentary_innings_2], ignore_index=True)

    return df_combined_commentary, df_wagon, df_statistics

def plot_manhattan_chart(df_statistics, placeholder):
    with placeholder.container():
        if not df_statistics.empty:
            plt.figure(figsize=(10, 5))
            plt.bar(df_statistics['over'], df_statistics['runs'], color='blue')
            plt.xlabel("Over")
            plt.ylabel("Runs")
            plt.title("Manhattan Chart")
            st.pyplot(plt)
        else:
            st.warning("No data available for Manhattan Chart.")

def plot_worm_chart(df_statistics, placeholder):
    with placeholder.container():
        if not df_statistics.empty:
            plt.figure(figsize=(10, 5))
            plt.plot(df_statistics['over'], df_statistics['cumulative_runs'], marker='o')
            plt.xlabel("Over")
            plt.ylabel("Cumulative Runs")
            plt.title("Worm Chart")
            st.pyplot(plt)
        else:
            st.warning("No data available for Worm Chart.")

def plot_runrate_chart(df_statistics, placeholder):
    with placeholder.container():
        if not df_statistics.empty:
            plt.figure(figsize=(10, 5))
            plt.plot(df_statistics['over'], df_statistics['runrate'], marker='o', color='green')
            plt.xlabel("Over")
            plt.ylabel("Run Rate")
            plt.title("Run Rate Chart")
            st.pyplot(plt)
        else:
            st.warning("No data available for Run Rate Chart.")

def plot_extras_chart(df_statistics, placeholder):
    with placeholder.container():
        extras = df_statistics.get("extras", [])
        if extras:
            extras_df = pd.DataFrame(extras)
            plt.figure(figsize=(10, 5))
            plt.bar(extras_df['name'], extras_df['value'], color='orange')
            plt.xlabel("Extra Types")
            plt.ylabel("Count")
            plt.title("Extras Distribution")
            st.pyplot(plt)
        else:
            st.warning("No data available for Extras Distribution.")

def main():
    st.title("Cricket Analyzer")

    match_ids = {
        "MI Cape Town vs Joburg Super Kings": "82622",
        "Match 2": "87107",
        "Match 3": "87108"
    }

    selected_match = st.selectbox("Select a Match", list(match_ids.keys()))
    match_id = match_ids[selected_match]

    if st.button("Fetch and Analyze Data"):
        with st.spinner("Fetching data..."):
            time.sleep(1)  # Simulate loading time
            all_data = fetch_and_store_all_data(match_id)
            df_commentary, df_wagon, df_statistics = create_dataframes(all_data)
            st.success("Data fetched and processed successfully!")

            st.subheader("Visualizations")

            chart_placeholder = st.empty()

            if st.button("Manhattan Chart"):
                with st.spinner("Loading Manhattan Chart..."):
                    time.sleep(1)  # Simulate loading time
                    plot_manhattan_chart(df_statistics, chart_placeholder)

            if st.button("Worm Chart"):
                with st.spinner("Loading Worm Chart..."):
                    time.sleep(1)  # Simulate loading time
                    plot_worm_chart(df_statistics, chart_placeholder)

            if st.button("Run Rate Chart"):
                with st.spinner("Loading Run Rate Chart..."):
                    time.sleep(1)  # Simulate loading time
                    plot_runrate_chart(df_statistics, chart_placeholder)

            if st.button("Extras Distribution"):
                with st.spinner("Loading Extras Distribution..."):
                    time.sleep(1)  # Simulate loading time
                    plot_extras_chart(df_statistics, chart_placeholder)

if __name__ == "__main__":
    main()
