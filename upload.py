import streamlit as st

def main():
    st.title("File Upload Tutorial")

    menu = ["Image","Dataset","DocumentFiles","About"]
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "Image":
        st.subheader("Image")

    elif choice == "Dataset":
        st.subheader("Dataset")

    elif choice == "DocumentFiles":
        st.subheader("DocumentFiles")