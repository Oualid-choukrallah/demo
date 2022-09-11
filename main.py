import streamlit as st
from deepface import DeepFace
from retinaface import RetinaFace
from PIL import Image
import os


def load_image(image_file):
	img = Image.open(image_file)
	return img

def save_uploadedfile(uploadedfile):
    with open(os.path.join("media", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Saved File:{} to media".format(uploadedfile.name))

st.set_page_config(layout="wide")
st.title("Demos")

logo = Image.open("canal plus logo.png")
deepface_logo = Image.open("deepface_logo.png")
with st.sidebar :
    st.image(logo)
    feature = st.radio("Choose your algorithm", ("👩/👨 DeepFace", "⌛ Coming soon"))
    #tab1, tab2 = st.tabs(["👩/👨 DeepFace"", "⌛ Coming soon"])

if feature == '👩/👨 DeepFace':
    st.markdown('##')
    title_container = st.container()
    col1, col2 = st.columns([1, 20])
    with title_container:
        with col1:
            st.image(deepface_logo)
        with col2:
            st.markdown("<h2 style='text-align: left; color: black;'>DeepFace</h2>", unsafe_allow_html=True)
    st.markdown('#')
    uploaded_file = st.file_uploader("Choose a file", type=["png","jpg","jpeg"])
    if uploaded_file is not None :
        container2 = st.container()
        c1, c2, c3 = st.columns([3,2,3])
        save_uploadedfile(uploaded_file)
        with container2:
            with c2 :
                st.image(load_image(uploaded_file), width=400)
        faces = RetinaFace.extract_faces(os.path.join("media", uploaded_file.name))

        #Prediction
        result = []
        i = 0
        for face in faces:
            predictions = DeepFace.analyze(face, actions=['gender'], enforce_detection=False, detector_backend='skip')
            result.append(predictions['gender'])
            i = i+1
            st.metric(label=f"Gender {i}", value=predictions['gender'])

        st.balloons()