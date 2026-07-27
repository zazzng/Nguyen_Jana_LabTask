import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="Vancouver Neighborhood Explorer",
    layout="wide",
)

PALETTE = ["#e8925a", "#4a7ba6", "#c1436d", "#5aa872", "#8a6bbf",
           "#d4a13c", "#4bb0c4", "#b0563a"]

st.title("Vancouver Neighborhood Similarity Explorer")
st.write("Which neighborhoods look similar based on their mix of business types?")


@st.cache_data
def load_data(path="vancouver_businesses.csv"):
    return pd.read_csv(path)


df = load_data()

MIN_BUSINESSES = 50


@st.cache_data
def build_composition(df):
    # count businesses per area, keep only areas above the cutoff
    area_counts = df["localarea"].value_counts()
    keep_areas = area_counts[area_counts >= MIN_BUSINESSES].index
    df_area = df[df["localarea"].isin(keep_areas)]

    # area x businesstype matrix, row-normalized to percentages
    composition = pd.crosstab(
        df_area["localarea"],
        df_area["businesstype"],
        normalize="index"
    ) * 100

    counts = area_counts[keep_areas]
    return composition, counts


composition, area_counts = build_composition(df)

with st.expander("Look at Data:"):
    st.dataframe(df.head(20))
    st.write(f"{len(df):,} businesses, {df['localarea'].nunique()} neighborhoods.")
    st.write(f"Composition matrix: {composition.shape[0]} areas x {composition.shape[1]} business types")
    st.dataframe(composition.round(1))

st.sidebar.header("1. Clustering")
k = st.sidebar.slider("Number of clusters (K)", 2, 8, 4)

X = composition.to_numpy()
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)

color_map = {str(c): PALETTE[c % len(PALETTE)] for c in range(k)}

col1, col2, col3 = st.columns(3)
col1.metric("Businesses", f"{len(df):,}")
col2.metric("Neighborhoods", composition.shape[0])
col3.metric("Clusters", k)

st.divider()

pca = PCA(n_components=2, random_state=42)
coords_2d = pca.fit_transform(X)
var = pca.explained_variance_ratio_

plot_df = pd.DataFrame({
    "area": composition.index,
    "PC1": coords_2d[:, 0],
    "PC2": coords_2d[:, 1],
    "cluster": labels.astype(str),
})

centroids = df.groupby("localarea")[["lat", "lon"]].mean()
map_df = pd.DataFrame({
    "area": composition.index,
    "cluster": labels.astype(str),
    "n_businesses": area_counts.loc[composition.index].to_numpy(),
})
map_df["lat"] = centroids.loc[map_df["area"], "lat"].to_numpy()
map_df["lon"] = centroids.loc[map_df["area"], "lon"].to_numpy()

dr_tab, map_tab = st.tabs(["Business Mix (PCA)", "Map"])

with dr_tab:
    st.write(f"PC1 and PC2 capture {var.sum():.0%} of the variation.")
    fig = px.scatter(
        plot_df, x="PC1", y="PC2",
        color="cluster",
        text="area",
        color_discrete_map=color_map,
        height=600,
    )
    fig.update_traces(textposition="top center", marker=dict(size=14))
    st.plotly_chart(fig, width="stretch")

with map_tab:
    fig_map = px.scatter_map(
        map_df,
        lat="lat", lon="lon",
        color="cluster",
        size="n_businesses",
        hover_name="area",
        color_discrete_map=color_map,
        zoom=11,
        height=600,
        map_style="carto-positron",
        center={"lat": map_df["lat"].mean(), "lon": map_df["lon"].mean()},
    )
    st.plotly_chart(fig_map, width="stretch")

st.divider()

st.subheader("Which neighborhoods are in each cluster")

members = pd.DataFrame({
    "area": composition.index,
    "cluster": labels,
    "n_businesses": area_counts.loc[composition.index].to_numpy(),
})

for c in sorted(members["cluster"].unique()):
    areas_in = members[members["cluster"] == c].sort_values("n_businesses", ascending=False)
    cluster_areas = areas_in["area"]
    top_types = composition.loc[cluster_areas].mean().sort_values(ascending=False).head(3)

    st.markdown(f"**Cluster {c}** ({len(areas_in)} areas)")
    st.write("Neighborhoods: " + ", ".join(areas_in["area"]))
    st.write("Top business types: " + ", ".join(
        f"{name} ({pct:.0f}%)" for name, pct in top_types.items()
    ))
    st.write("")
