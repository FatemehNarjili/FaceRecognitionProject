import streamlit as st
import os


current_working_directory = os.getcwd()
# _____________________header_______________________________
st.header('Our Model:', divider='rainbow')
# _______________detection_____________________________
st.subheader('Detection', divider='green')
st.write(':red[**MTCNN**]')
st.markdown(u'\u2022 Multi-Task Cascaded Convolutional Networks')
st.markdown(u'\u2022 uses a cascading series of convolutional neural networks (CNNs) to detect and localize faces in digital images or videos.')
st.image(current_working_directory + '/pictures/mtcnn.png',
         caption='mtcnn structure')
# __________________________embeding___________________________
st.subheader('Embeding', divider='green')
st.write(':red[**AdaFAce**]')
st.markdown(
    u'\u2022 was introduced at the CVPR (Conference on Computer Vision and Pattern Recognition) in 2022.')
st.markdown(u'\u2022 Its main contribution lies in the introduction of image quality as a new variable for the adaptive loss function.')
st.markdown(
    u'\u2022 AdaFace modifies the loss function to improve face recognition performance:')
st.divider()
st.image(current_working_directory + '/pictures/adaface1.png',
         caption='easy vs hard images')
st.image(current_working_directory + '/pictures/adaface2.png',
         caption='Equal Emphasis')
st.image(current_working_directory + '/pictures/adaface3.png',
         caption='Hard Mining')
st.image(current_working_directory + '/pictures/adaface4.png',
         caption='Proposed AdaFace')
st.image(current_working_directory + '/pictures/adaface5.png',
         caption='Final Goal')
st.subheader(':blue[**How does this works?**]')
st.image(current_working_directory + '/pictures/AdaptiveMargin.png',
         caption='Adaptive Margin')
st.image(current_working_directory + '/pictures/adaface_demo5.gif',
         caption='ArcFace vs AdaFace')
