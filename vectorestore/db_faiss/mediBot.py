import streamlit as st


def main():
    st.title("Ask MediBot!")

    Prompt=st.chat_input("Pass your prompt here:")

    if Prompt:
        st.chat_message("user").markdown(Prompt)

        response= "Hello! How may I help you today?"

        st.chat_message("assistant").markdown(response)

if __name__ == "__main__":
    main()





