import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

st.title('Market Basket Analysis of Bakery Products')
st.write('Upload Bakery Products for Association Rule')

# file upload
uploaded_file = st.file_uploader("upload file",type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Raw Data")
    st.write(df.head())

    i=df['Item'].tolist()
    df=df.drop(df.loc[df['Item'] == 'NONE'].index)
    df.loc[df['Item']=='NONE'].count()
    count = df['Item'].value_counts()
    df['Item'].unique().tolist()

    basket_items={}
    for item in df['Item']:
        if item in basket_items:
            basket_items[item] = basket_items[item] + 1
        else:
            basket_items[item] = 1
    basket_items

    item_names = []
    item_freq = []
    for key,value in basket_items.items():
        item_names.append(key)
        item_freq.append(value)

    df = df.groupby('Transaction').agg(','.join).reset_index()
    df = df.drop(['Date','Time'], axis=1)
    items_data=df['Item']

    items_list = [item.split(',') for item in items_data]
    item_list_df = pd.DataFrame({'Items':items_list})

    t = TransactionEncoder()
    t_array = t.fit(items_list).transform(items_list)
    df_encoded = pd.DataFrame(t_array, columns = t.columns_)
    
    # Apply Apriori
    frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)

    # Generate rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)

    st.subheader("📈 Association Rules")
    st.write(rules.head())

    # ---------------------------
    # 📊 Scatter Plot
    # ---------------------------
    st.subheader("📊 Support vs Confidence (Lift as Color)")

    fig1, ax1 = plt.subplots()
    scatter = ax1.scatter(
        rules['support'],
        rules['confidence'],
        c=rules['lift']
    )
    ax1.set_xlabel("Support")
    ax1.set_ylabel("Confidence")
    fig1.colorbar(scatter, label="Lift")
    st.pyplot(fig1)

    # ---------------------------
    # 🔥 Heatmap
    # ---------------------------
    st.subheader("🔥 Lift Heatmap")

    pivot = rules.pivot_table(
        index='antecedents',
        columns='consequents',
        values='lift'
    )

    fig2, ax2 = plt.subplots(figsize=(10,6))
    sns.heatmap(pivot, annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

    # ---------------------------
    # 🌐 Network Graph
    # ---------------------------
    st.subheader("🌐 Association Rule Network")

    G = nx.DiGraph()

    for _, row in rules.iterrows():
        for ant in row['antecedents']:
            for con in row['consequents']:
                G.add_edge(ant, con, weight=row['lift'])

    pos = nx.spring_layout(G)

    fig3, ax3 = plt.subplots(figsize=(10,8))
    nx.draw(
        G, pos,
        with_labels=True,
        node_color='lightblue',
        node_size=2000,
        font_size=10
    )

    st.pyplot(fig3)

    # ---------------------------
    # 🎯 Filter Rules
    # ---------------------------
    st.subheader("🎯 Filter Strong Rules")

    min_lift = st.slider("Minimum Lift", 1.0, 3.0, 1.2)
    min_conf = st.slider("Minimum Confidence", 0.0, 1.0, 0.4)

    filtered_rules = rules[
        (rules['lift'] >= min_lift) &
        (rules['confidence'] >= min_conf)
    ]

    st.write(filtered_rules)

else:
    st.info("Please upload a CSV file to proceed.")
    
    

    