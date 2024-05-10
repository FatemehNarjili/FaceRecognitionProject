import streamlit as st
import os


current_working_directory = os.getcwd()

st.header('Libraries:', divider='rainbow')

# _______________DeepFace_______________________________________________________________________________________
st.subheader(':red[1.DeepFace]')
st.markdown(u'\u2022 lightweight face recognition and facial attribute analysis (age, gender, emotion and race) framework for python.')
st.markdown(u"\u2022 developed by Facebook's AI Research (FAIR) lab in 2014.")
st.markdown(
    u'\u2022 accuracy rate of approximately 97.35% on the Labeled Faces in the Wild (LFW) dataset.')


st.caption('Verify')
code = '''models = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
  "GhostFaceNet",
]


result = DeepFace.verify(img1_path = "img1.jpg", 
      img2_path = "img2.jpg", 
      model_name = models[0]
)
'''
st.code(code, language='python')


st.image(current_working_directory + '/pictures/df-detectors.png',
         caption='Different Models')
st.image(current_working_directory +
         '/pictures/df-verify.png', caption='verify')

st.caption('find')
code = '''dfs = DeepFace.find(img_path = "img1.jpg",
      db_path = "C:/workspace/my_db", 
      model_name = models[1]
)'''
st.code(code, language='python')
st.image(current_working_directory + '/pictures/df-find.png', caption='find')
st.divider()

# _______________Face-Recognition_________________________________________
st.subheader(':red[2.Face-Recognition]')
st.markdown(u'\u2022 Used for Recognize and manipulate faces.')
st.markdown(u'\u2022 was released in 2017 by Adam Geitgey.')
st.markdown(
    u'\u2022 Built using dlib’s(dlib c++ library) state-of-the-art face recognition.')
st.markdown(
    u'\u2022 has an accuracy of 99.38% on the Labeled Faces in the Wild (LFW) dataset.')


st.caption('Detection')
code = '''face_recognition.api.batch_face_locations(images,/
number_of_times_to_upsample=1, batch_size=128)'''
st.code(code, language='python')

st.caption('Embedings')
code = '''face_recognition.api.face_encodings(face_image,/
known_face_locations=None, num_jitters=1, model='small')'''
st.code(code, language='python')

st.caption('Compare')
code = '''face_recognition.api.compare_faces(known_face_encodings,/
face_encoding_to_check, tolerance=0.6)'''
st.code(code, language='python')
st.caption('Example:')
st.image(current_working_directory + '/pictures/biden.png',
         caption='Recognizing biden')
code = '''known_image = face_recognition.load_image_file("biden.jpg")
unknown_image = face_recognition.load_image_file("unknown.jpg")

biden_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

results = face_recognition.compare_faces([biden_encoding], unknown_encoding))'''
st.code(code, language='python')

st.divider()

# ______________________________state-of-the-art___________________________________________________________

st.header('State of the Art:', divider='rainbow')

st.image(current_working_directory + '/pictures/paperswithcode.png',
         caption='state of the art')

st.subheader(':red[AdaFace]')
st.image(current_working_directory +
         '/pictures/adaface.png', caption='adaface')
