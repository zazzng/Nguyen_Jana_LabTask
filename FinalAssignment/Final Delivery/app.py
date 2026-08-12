import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans


st.set_page_config(page_title="Pixel Publishing Success Screener", layout="wide")
ACCENT = "#e8925a"

genres = ['Indie', 'Casual', 'Action', 'Adventure', 'Simulation', 'Strategy', 'RPG', 'Free To Play', 'Early Access', 'Sports', 'Racing', 'Massively Multiplayer']
nongame = ['Utilities', 'Design & Illustration', 'Animation & Modeling', 'Video Production', 'Game Development', 'Audio Production', 'Software Training', 'Photo Editing', 'Web Publishing', 'Accounting', 'Movie', 'Documentary', 'Short', '360 Video', 'Tutorial','Episodic']
col_names = ['AppID', 'Name', 'Release date', 'Estimated owners', 'Peak CCU', 'Required age', 'Price', 'Discount', 'DLC count', 'About the game', 'Supported languages', 'Full audio languages', 'Reviews', 'Header image', 'Website', 'Support url', 'Support email', 'Windows', 'Mac', 'Linux', 'Metacritic score', 'Metacritic url', 'User score', 'Positive', 'Negative', 'Score rank', 'Achievements', 'Recommendations', 'Notes', 'Average playtime forever', 'Average playtime two weeks', 'Median playtime forever', 'Median playtime two weeks', 'Developers', 'Publishers', 'Categories', 'Genres', 'Tags', 'Screenshots', 'Movies']

@st.cache_data
def load():
    df = pd.read_csv("games.csv", skiprows=1, names=col_names, index_col=False, low_memory=False)
    lower = df['Estimated owners'].str.split(' - ').str[0].astype(int)
    df['success'] = (lower >= 50000).astype(int)

    tok = df['Genres'].fillna('').apply(lambda s: [x.strip() for x in s.split(',')])
    has_nongame = tok.apply(lambda t: any(x in nongame for x in t))
    has_game = tok.apply(lambda t: any(x in genres for x in t))
    is_addon = df['Name'].fillna('').str.contains(r'\bsoundtrack\b|\bost\b|\bdemo\b|\bart\s?book\b', case=False, regex=True)
    df = df[~((has_nongame & ~has_game) | is_addon)].copy()

    gd = df['Genres'].str.get_dummies(sep=',')
    X = gd[genres].copy()
    X['plat_mac'] = df['Mac'].astype(int)
    X['plat_linux'] = df['Linux'].astype(int)
    X['age_restricted'] = (df['Required age'] > 0).astype(int)
    X['price_log'] = np.log1p(df['Price'])
    n = df['Supported languages'].fillna('').str.count(',') + 1
    n[df['Supported languages'].fillna('').str.strip().isin(['', '[]'])] = 0
    X['n_languages_log'] = np.log1p(n)
    X['release_month'] = pd.to_datetime(df['Release date'], errors='coerce').dt.month.fillna(0).astype(int)
    X['is_self_published'] = (((df['Developers'].fillna('') == df['Publishers'].fillna('')) & df['Developers'].notna()).astype(int))
    y = df['success']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42).fit(scaler.transform(X_train), y_train)
    auc = roc_auc_score(y_test, model.predict_proba(scaler.transform(X_test))[:, 1])
    all_probs = model.predict_proba(scaler.transform(X))[:, 1]

    text_df = df[df['About the game'].notna()].copy()
    desc = text_df['About the game'].str.replace(r'<[^>]+>', ' ', regex=True)
    tfidf = TfidfVectorizer(max_features=800, stop_words='english', min_df=20)
    Xtext = tfidf.fit_transform(desc)
    Xsvd = TruncatedSVD(n_components=100, random_state=42).fit_transform(Xtext)
    text_df['theme'] = KMeans(n_clusters=5, random_state=42, n_init=5).fit_predict(Xsvd)

    terms = np.array(tfidf.get_feature_names_out())
    labels = text_df['theme'].values
    rows = []
    for c in range(5):
        mask =labels == c
        top = terms[np.asarray(Xtext[mask].mean(axis=0)).ravel().argsort()[::-1][:3]]
        rows.append({'theme': ', '.join(top), 'games': int(mask.sum()), 'success_rate': text_df['success'].values[mask].mean()})
    themes = pd.DataFrame(rows).sort_values('success_rate', ascending=False)
    return df, X, y, model, scaler, auc, all_probs,themes

df, X, y, model, scaler, auc, all_probs, themes = load()
base_rate = y.mean()

st.title("Pixel Publishing - Game Success Screener")
st.caption(f"Success = a game reaching 50,000+ owners.  Model AUC = {auc:.2f}.")

tab1, tab2, tab3 = st.tabs(["Score a concept", "Explore", "Description themes"])
with tab1:
    st.subheader("Score a game concept")
    pick = st.multiselect("Genres", genres, default=["Action"])
    price = st.slider("Price (USD)", 0.0, 70.0, 15.0)
    n_lang = st.slider("Number of languages", 1, 30, 5)
    month = st.slider("Release month", 1, 12, 10)
    mac = st.checkbox("On Mac")
    linux = st.checkbox("On Linux")
    age = st.checkbox("Age rating (17+)")
    selfpub = st.checkbox("Self-published")

    row = {}
    for g in genres:
        row[g] = 1 if g in pick else 0
    row['plat_mac'] = int(mac)
    row['plat_linux'] = int(linux)
    row['age_restricted'] = int(age)
    row['price_log'] = np.log1p(price)
    row['n_languages_log'] = np.log1p(n_lang)
    row['release_month'] = month
    row['is_self_published'] = int(selfpub)
    row = pd.DataFrame([row])[X.columns]

    prob = model.predict_proba(scaler.transform(row))[0, 1]
    score = (all_probs < prob).mean() * 100

    st.metric("Screening score", f"{score:.0f} / 100")
    if score >= 75:
        st.success("Strong concept, ranks in the top quarter.")
    elif score >= 50:
        st.info("Above average, ranks in the top half.")
    else:
        st.warning("Below average on these traits.")
    st.caption("This is a relative ranking to help prioritise concepts, not a guarantee." "It shows associations in the data, not causes.")

with tab2:
    st.subheader("Success rate by genre")
    rate= pd.Series({g: y[X[g] == 1].mean() for g in genres}).sort_values(ascending=False)
    pick_g = st.selectbox("Pick a genre", rate.index)
    st.metric(f"{pick_g} success rate", f"{rate[pick_g]:.1%}",
              delta=f"{(rate[pick_g] - base_rate) * 100:+.1f} pts vs overall")
    st.bar_chart(rate)

    st.subheader("Success rate by price")
    price_band = pd.cut(df['Price'], bins = [-0.01, 0, 5, 15, 30, 1000], labels = ['Free', '0-5', '5-15', '15-30', '30+'])
    st.bar_chart(df.groupby(price_band, observed=True)['success'].mean())

    st.subheader("Success rate by number of languages")
    n_lang= df['Supported languages'].fillna('').str.count(',') + 1
    lang_band =pd.cut(n_lang, bins=[0, 1, 3, 6, 10, 100], labels = ['1', '2-3', '4-6', '7-10', '11+'])
    st.bar_chart(df.groupby(lang_band, observed=True)['success'].mean())

    st.subheader("Success rate by release month")
    month = pd.to_datetime(df['Release date'], errors = 'coerce').dt.month
    st.line_chart(df.groupby(month)['success'].mean())

with tab3:
    st.subheader("What kinds of games are out there")
    st.caption("I grouped the game descriptions into 5 themes with K-means (unsupervised).""Each theme is labelled by its top words. Bars show how often each theme reaches 50,000+ owners.")
    st.bar_chart(themes.set_index('theme')['success_rate'])
    st.dataframe(themes, hide_index=True)