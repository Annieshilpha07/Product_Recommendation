import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

# === 1. Load & preprocess ===
@st.cache_data
def load_data(path="fashion_combined_by_parent_asin.csv"):
    df = pd.read_csv(path)
    # ratings
    df_r = df[['user_id', 'asin', 'rating']].dropna()
    df_r.columns = ['user_id','item_id','rating']
    df_r = df_r.astype({'user_id':'category','item_id':'category','rating':'float32'})
    train, test = train_test_split(df_r, test_size=0.2, random_state=42)
    # mappings
    user_to_idx = {u:i for i,u in enumerate(train['user_id'].cat.categories)}
    item_to_idx = {i:j for j,i in enumerate(train['item_id'].cat.categories)}
    train['user_idx'] = train['user_id'].map(user_to_idx)
    train['item_idx'] = train['item_id'].map(item_to_idx)
    # sparse matrix
    R = csr_matrix(
        (train['rating'], (train['user_idx'], train['item_idx'])),
        shape=(len(user_to_idx), len(item_to_idx))
    )
    # content
    df['combined_text'] = df[['title_y','features','description']].fillna('').agg(' '.join,axis=1)
    df_items = df.drop_duplicates('asin')[['asin','combined_text']].reset_index(drop=True)
    return df, df_items, train, test, user_to_idx, item_to_idx, R

# === 2. Build CBF model ===
@st.cache_resource
def build_cbf(df_items):
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    M = tfidf.fit_transform(df_items['combined_text'])
    nn = NearestNeighbors(metric='cosine', algorithm='brute').fit(M)
    id2idx = {asin:i for i,asin in enumerate(df_items['asin'])}
    idx2id = {i:asin for asin,i in id2idx.items()}
    return M, nn, id2idx, idx2id

# === 3. CF helpers ===
def cf_sim_matrices(R):
    U = cosine_similarity(R)        # user×user
    I = cosine_similarity(R.T)      # item×item
    return U, I

def predict_cf_user(U, R, u, i, k=20):
    sims = U[u]
    top = np.argsort(sims)[-k:]
    r = R[top,i].toarray().flatten()
    s = sims[top]
    m = r>0
    return np.dot(r[m],s[m]) / (s[m].sum()+1e-8) if m.sum() else 0

def predict_cf_item(I, R, u, i, k=20):
    sims = I[i]
    top = np.argsort(sims)[-k:]
    r = R[u,top].toarray().flatten()
    s = sims[top]
    m = r>0
    return np.dot(r[m],s[m]) / (s[m].sum()+1e-8) if m.sum() else 0

# === 4. Hybrid recommender ===
def hybrid_recommendations(user_id, top_n, alpha,
                           train, user_to_idx, item_to_idx, R,
                           M, nn, id2idx, idx2id):

    uid = user_to_idx[user_id]
    seen = set(train[train['user_idx']==uid]['item_id'])
    U, I = cf_sim_matrices(R)

    # precompute CBF scores: for each candidate item, average hits in top-20 neighbors of seen items
    cbf = {}
    for asin in idx2id.values():
        iidx = id2idx[asin]
        neigh = nn.kneighbors(M[iidx], n_neighbors=20+1)[1].flatten()[1:]
        hits = sum(idx2id[n] in seen for n in neigh)
        cbf[asin] = hits / len(seen) if seen else 0

    # compute final scores
    scores = []
    for asin,iidx in id2idx.items():
        if asin in seen: continue
        cf_u = predict_cf_user(U, R, uid, item_to_idx.get(asin, -1))
        cf_i = predict_cf_item(I, R, uid, item_to_idx.get(asin, -1))
        cf = 0.5*(cf_u+cf_i)
        sc = alpha*cf + (1-alpha)*cbf[asin]
        scores.append((asin, sc))

    top = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
    return [a for a,_ in top]

# === 5. Streamlit UI ===
def main():
    st.title("Hybrid Product Recommender")

    df, df_items, train, test, user_to_idx, item_to_idx, R = load_data()
    M, nn, id2idx, idx2id = build_cbf(df_items)

    st.sidebar.header("Settings")
    user = st.sidebar.selectbox("User ID", options=list(user_to_idx.keys()))
    alpha = st.sidebar.slider("CF vs CBF (α)", 0.0, 1.0, 0.5)
    n = st.sidebar.number_input("Top N", value=10, min_value=1, max_value=50)

    if st.sidebar.button("Recommend"):
        with st.spinner("Computing..."):
            recs = hybrid_recommendations(user, n, alpha,
                                          train, user_to_idx, item_to_idx, R,
                                          M, nn, id2idx, idx2id)
        st.subheader(f"Recommendations for {user}")
        for asin in recs:
            txt = df_items[df_items['asin']==asin]['combined_text'].iloc[0][:80]
            st.markdown(f"**{asin}** — {txt}...")

if __name__=="__main__":
    main()
