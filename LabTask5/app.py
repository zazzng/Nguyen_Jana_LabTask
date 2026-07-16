import streamlit as st
import json
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("running")

st.title("Business Location Explorer")
st.write("Learning how to use streamlit")

#TODO: cache a function and data

def load_data(path="business_locations.geojson"):
    with open(path) as f:
        geojson = json.load(f)
    rows = []
    #this function can read geojson files that have only points.
    for feat in geojson["features"]:
        props = feat["properties"]
        lon, lat = feat["geometry"]["coordinates"]
        rows.append({**props, "lon":lon, "lat":lat})
    return pd.DataFrame(rows)

df = load_data()

with st.expander("Look at Data:"):
    st.dataframe(df.head(20))
    st.write(f"{len(df)} locations, {df["Neighborhood"].nunique()} neighborhoods.")

st.sidebar.header("1. Select Features")



NUMERIC_COLS = ["Floor_Area_sqm","Daily_Foot_Traffic","Community_Impact_Score","Annual_Revenue_k"]
selected_features = st.sidebar.multiselect("Features to be used in models", options=NUMERIC_COLS,default=NUMERIC_COLS)

if len(selected_features) < 2:
    st.warning("Pick at least two features to continue")
    st.stop()

X = df[selected_features].to_numpy()
X_scaled = StandardScaler().fit_transform(X)

st.sidebar.header("2. Clustering")
algo= st.sidebar.selectbox('Algorithm', ["Kmeans","DBSCAN"])
if algo == "Kmeans":
    k = st.sidebar.slider("Number of Clusters",2,10)
elif algo == "DBSCAN":
    eps = st.sidebar.slider("eps (neighborhood radius)", 0.1, 2.0, 0.5, 0.05)
    min_samples = st.sidebar.slider("min_samples", 2, 20, 5)

labels = []
if algo == "Kmeans":
    model = KMeans(n_clusters=k)
    labels = model.fit_predict(X_scaled)
elif algo == "DBSCAN":
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)

# Catch error if no labels do not continue
if  len(labels) ==0 : 
    st.warning("there is no clustering labels")
    st.stop()

df["cluster"] = pd.Categorical(labels.astype(str))
n_clusters_found = df["cluster"].nunique()
st.metric("Number of Clusters", n_clusters_found)

map_tab, dr_tab = st.tabs(["Map","Dimensionality Reduction"])

with map_tab:
    st.write("MAP")
    fig = px.scatter_map(df, lat="lat", lon="lon", color="cluster", zoom=10, height=550, 
        map_style="carto-darkmatter")
    st.plotly_chart(fig, width="stretch")

with dr_tab:
    reducer = PCA(n_components=2,random_state=42)
    embedding = reducer.fit_transform(X_scaled)
    df["dim_1"] = embedding[:,0]
    df["dim_2"] = embedding[:,1]

    fig_dr  = px.scatter(
        df, 
        x="dim_1",
        y="dim_2",
        color="cluster",
        height = 550
    )
    st.plotly_chart(fig_dr,width="stretch")