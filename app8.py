import streamlit as st
import pandas as pd
import os

from langchain_groq.chat_models import ChatGroq
from langchain_experimental.agents.agent_toolkits.pandas.base import (
    create_pandas_dataframe_agent
)

# ----------------------------------
# Streamlit Page Configuration
# ----------------------------------

st.set_page_config(
    page_title="Enterprise AI CSV Assistant",
    page_icon="📊",
    layout="wide"
)

# ----------------------------------
# Title
# ----------------------------------

st.title("📊 Enterprise AI CSV Data Assistant")
st.markdown("Upload CSV datasets and query using AI")

# ----------------------------------
# Sidebar
# ----------------------------------

st.sidebar.header("⚙ Configuration")

groq_api_key = st.sidebar.text_input(
    "Enter Groq API Key",
    type="password"
)

model_name = st.sidebar.selectbox(
    "Select Groq Model",
    [
        "llama-3.1-8b-instant",
        "llama3-70b-8192"
    ]
)

# ----------------------------------
# File Upload
# ----------------------------------

uploaded_file = st.file_uploader(
    "📂 Upload CSV Dataset",
    type=["csv"]
)

# ----------------------------------
# Main Workflow
# ----------------------------------

if uploaded_file is not None:

    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)

        st.success("✅ Dataset Uploaded Successfully")

        # Show Dataset
        with st.expander("📄 Preview Dataset"):
            st.dataframe(df.head())

        # Dataset Info
        col1, col2, col3 = st.columns(3)

        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", df.isnull().sum().sum())

        # ----------------------------------
        # AI Query Section
        # ----------------------------------

        st.subheader("🤖 Ask Questions About Your Data")

        user_query = st.text_area(
            "Enter your question",
            placeholder="Example: Which student has highest accuracy?"
        )

        # ----------------------------------
        # Generate Answer
        # ----------------------------------

        if st.button("Generate AI Answer"):

            if not groq_api_key:
                st.error("Please enter Groq API Key")
            else:

                with st.spinner("AI is analyzing dataset..."):

                    try:

                        # Initialize Groq LLM
                        llm = ChatGroq(
                            model_name=model_name,
                            api_key=groq_api_key
                        )

                        # Create Pandas Agent
                        agent = create_pandas_dataframe_agent(
                            llm,
                            df,
                            verbose=True,
                            allow_dangerous_code=True
                        )

                        # Run Query
                        response = agent.run(user_query)

                        # Display Output
                        st.subheader("✅ AI Response")
                        st.success(response)

                    except Exception as e:
                        st.error(f"AI Error: {e}")

    except Exception as e:
        st.error(f"Dataset Error: {e}")

else:
    st.info("Upload a CSV file to begin")

# ----------------------------------
# Footer
# ----------------------------------

st.markdown("---")
st.caption("Enterprise AI CSV Assistant using Streamlit + LangChain + Groq")
