import streamlit as st
import pandas as pd
from pandasai import Agent
from pandasai.llm.openai import OpenAI

st.set_page_config(
    page_title="Arbitrum Grants Chatbot",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "Get Help": None,
        "Report a bug": "https://twitter.com/sageOlamide",
        "About": None
    }
)

st.title("Arbitrum Grants Chatbot")

with st.expander("About"):
    st.write("Chat with a dataset of Arbitrum grants scraped from [Karma GAP](https://gap.karmahq.xyz/arbitrum).")
    st.write("TIP: if you are having trouble getting the information you need, try refining your question using one or more of the dataset column names: `grantee`, `grant_date`, `grant_amount_arb`, `grant_name`, `proposal_url`, `gap_url`.")
    st.write("All amounts are denominated in ARB.")
# load dataset
dataset = pd.read_csv("arbitrum_grantees.csv")
# Instantiate a LLM
llm = OpenAI(api_token=st.secrets.api_key)
df = Agent(dataset, config={"llm": llm})

with st.form("Question"):
  question = st.text_area("Question", value="What are the top 5 grantees by amount received, and how much did they receive?")
  submitted = st.form_submit_button("Submit")
  if submitted:
    with st.spinner("Thinking..."):
      response = df.chat(question)
      if response is not None:
        if isinstance(response, str):
          response = response.replace('$', '')
          if response.endswith('.'):
            response = response[:-1]
        st.write(response)
        
with st.expander("View dataset"):
  st.dataframe(dataset)
      
st.download_button(
    label="Download dataset as CSV",
    data=dataset.to_csv().encode('utf-8'),
    file_name='arbitrum_grantees.csv',
    mime='text/csv',
)
