# from urllib.error import URLError
import pandas as pd
import altair as alt
import streamlit as st
from PIL import Image
from pathlib import Path

@st.cache
def get_path():
    return Path(__file__).parents[0]

@st.cache
def get_reg_results():
    df_reg = pd.read_csv(get_path() / 'data/df_reg.csv')
    return df_reg

@st.cache
def get_seq_results():
    df_seq = pd.read_csv(get_path() / 'data/df_seq.csv')
    return df_seq

@st.cache
def get_user_df():
    df_user = pd.read_csv(get_path() / 'data/user.csv')
    return df_user

@st.cache
def get_user_history_df():
    df_user = pd.read_csv(get_path() / 'data/user_hist.csv')
    return df_user

@st.cache
def get_main_categories(df_reg):
    cols = df_reg.columns
    print(len(cols), cols[2])
    return sorted(cols[2:])

st.set_page_config(layout='wide') # must be the first streamlit command

st.sidebar.image(Image.open(get_path() / 'logo.png'), use_column_width=False)
st.title('Top K Dashboard')
st.sidebar.title("Amazon Reviews Dataset")

with st.sidebar:
    dict_ds = {'# reviews': 5_613_183, '# users': 830_668, '# categories under Electronics': 36}
    df = pd.DataFrame(dict_ds, index=[0]).T
    df.columns = ['Count']
    s = df.style.format({"Count": lambda x : '{:,.0f}'.format(x)})
    st.dataframe(s)

st.sidebar.title("Modeling")
with st.sidebar:
    st.subheader("Overall performance using pairwise metrics NDCG@10, Hit@10")
    dict_ds = {'type':['Collaborative filtering', ' ', 'Content-based filtering', 'Hybrid', ' '],
            'name': ['SLi-Rec', 'SASRec', 'LightGBM', 'Wide & Deep', 'xDeepFM'],
            'ndcg@10': [0.404, 0.392, 0.0725, 0.1256, 0.1881],
            'hit@10': [0.6654, 0.628, 0.1631, 0.2781, 0.3497]}
    df = pd.DataFrame(dict_ds)
    df.set_index(['type', 'name'], inplace=True)
    df.style.set_properties(subset=['type'], **{'width': '30px'})
    st.dataframe(df)

    st.markdown("**Regression**: LightGBM, Wide & Deep, xDeepFM")
    st.markdown("**Binary classification**: SLi-Rec, SASRec")

st.markdown("""
This app shows the top **`k`** results by ensembling results from 5 recommendation system models on a subset of the popular [Amazon Reviews](http://deepyeti.ucsd.edu/jianmo/amazon/index.html) dataset. The dataset contains items that belong to 12 main categories (and an additional 24 sub-categories) of items that fall under the umbrella of **Electronics**. More information on the models and the dataset is available [here](https://github.com/ss-github-code/capstone_recsys/blob/main/report/report.md).
    
""")
st.markdown("""
The 5 models fall under 2 problem types: **regression** and **binary classification**. In regression, the models are predicting the target rating given to an unseen item by a user from the dataset. In binary classification, the models are predicting the probability of an item being reviewed next by a user from the dataset. For this dashboard, we chose the user with the most reviews in our dataset. Let's look at how the models predict the top k items among the selected categories.     
""")

k = st.slider("Choose k", min_value=5, max_value=25, value=10)

df_reg = get_reg_results()
df_seq = get_seq_results()
categories = st.multiselect(
    "Choose categories", list(get_main_categories(df_reg)), ["Amazon Devices", "Apple Products"]
)
if not categories:
    st.error("Please select at least one category.")
else:
    query_seq = ''
    query_reg = ''
    for i, c in enumerate(categories):
        query_seq += f'(df_seq["{c}"] == 1)'
        query_reg += f'(df_reg["{c}"] == 1)'
        if i != len(categories)-1:
            query_seq += ' | '
            query_reg += ' | '
    
    sel_df_reg = df_reg[eval(query_reg)][["title"] + categories]
    sel_df_reg = sel_df_reg.iloc[:k]
    sel_df_reg.reset_index(inplace=True, drop=True)
    sel_df_reg.reset_index(inplace=True)
    sel_df_reg['index'] = sel_df_reg['index'] + 1

    sel_df_seq = df_seq[eval(query_seq)][["title"] + categories]
    sel_df_seq = sel_df_seq.iloc[:k]
    sel_df_seq.reset_index(inplace=True, drop=True)
    sel_df_seq.reset_index(inplace=True)
    sel_df_seq['index'] = sel_df_seq['index'] + 1

    df_output_seq = sel_df_seq.melt(id_vars=['index','title'], 
                                    value_vars=categories, var_name='category')
    df_output_reg = sel_df_reg.melt(id_vars=['index','title'], 
                                    value_vars=categories, var_name='category')
    n = len(categories)
    w = n*25
    h = 30*k
    circles_seq = alt.Chart(df_output_seq).mark_circle().encode(
        x=alt.X('category:N',axis=alt.Axis(title=None, labelColor='black')),
        y=alt.Y('title:O', sort=None, axis=alt.Axis(title=None, grid=False)),
        size=alt.Size('value:Q', legend=None),
        color='category:N'
    ).properties(
        width=w,
        height=h,
        title='Sequential Models'
    )

    circles_reg = alt.Chart(df_output_reg).mark_circle().encode(
        x=alt.X('category:N',axis=alt.Axis(title=None, labelColor='black')),
        y=alt.Y('title:O', sort=None, axis=alt.Axis(title=None, grid=False)),
        size=alt.Size('value:Q', legend=None),
        color='category:N'
    ).properties(
        width=w,
        height=h,
        title='Regression Models'
    )
    middle = alt.Chart(sel_df_seq).encode(
        y=alt.Y('index:O', axis=None),
        text=alt.Text('index:Q')
    ).mark_text().properties(
        width=20,
        height=h
    )

    st.altair_chart(circles_reg | middle | circles_seq, use_container_width=True)

    st.subheader("Visualizing ratings made by the user with most reviews")
    st.markdown("""
    - The bar chart below shows the number of reviews made by this user in the selected categories.
    - The average rating given by the user is shown to the right of every bar.
    - Note that the ratings matrix is sparse! In addition, this user gives high ratings to every category!
    """)
    df_user = get_user_df()
    df_user = df_user[df_user['category'].isin(categories)]
    m = df_user['count'].max() + 5
    bars = alt.Chart(df_user).mark_bar().encode(
        x=alt.X('count:Q', axis=alt.Axis(title='Count', grid=False), scale=alt.Scale(domain=[0, m])),
        y=alt.Y('category:O', sort='-x', axis=alt.Axis(title=None, grid=False)),
        color=alt.Color('category:N')
    )
    text = bars.mark_text(
        align='left',
        baseline='middle',
        dx=3  # Nudges text to right so it doesn't appear on top of the bar
    ).encode(
        text='avg rating:Q'
    )
    st.altair_chart(bars+text, use_container_width=True)

    st.markdown("""Finally, let's also look at this history of reviews. 
    - The sequential models: SLi-Rec and SASRec have mechanisms to learn user behavior from the user's history of reviews.
    - Note the most recent reviews by this user have come in "Computers", "All Electronics", and "Home Audio & Theater categories"
    """)

    user_hist_df = get_user_history_df()
    timeline = alt.Chart(user_hist_df).mark_circle().encode(
        alt.X('date:T', axis=alt.Axis(title=None,labelAngle=0, format='%b, %Y')),
        alt.Y('main_cat:N', axis=alt.Axis(title=None, grid=False)),
        alt.Color('main_cat:N', legend=None)
    )
    st.altair_chart(timeline, use_container_width=True)