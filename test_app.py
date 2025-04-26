import streamlit as st

st.set_page_config(
    page_title="Test App",
    page_icon="🔄",
    layout="wide"
)

st.title("Kuramoto Model Test App")
st.write("If you can see this, Streamlit is working correctly!")

# Add a simple button and slider to test interactivity
value = st.slider("Test slider", 0, 100, 50)
st.write(f"Slider value: {value}")

if st.button("Test Button"):
    st.success("Button clicked!")