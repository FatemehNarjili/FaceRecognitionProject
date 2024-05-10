import streamlit as st

st.header('Pre-requirements')

st.caption('Database')
code = '''known_faces = defaultdict(
    lambda: {"embedding": None, "present": False, "count": 0})
current_ids = set()
absent_ids = set(list(known_faces.keys())) - current_ids'''
st.code(code, language='python')

st.caption('Getting Live Frames')
code = '''video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        continue'''
st.code(code, language='python')


st.caption('Models')
code = '''adaface_models = {
    'ir_50':"pretrained/adaface_ir50_ms1mv2.ckpt",
    'ir_101':"pretrained/adaface_ir101_ms1mv2.ckpt"
}

mtcnn = MTCNN()
model = load_pretrained_model('ir_50')'''
st.code(code, language='python')

st.subheader('Main Loop')
code = '''while True:
    ret, frame = video_capture.read()
    if not ret:
        continue

   
    boxes, faces = get_aligned_face(frame)
    if boxes is not None and faces is not None:

        for box, face in zip(boxes, faces):
            bgr_input = to_input(face)
            embeddings, _ = model(bgr_input)

            left, top, right, bottom = [int(coord) for coord in box]
    
            face_area = (right-left) * (bottom-top)
            frame_width, frame_height, _ = frame.shape
            frame_area = frame_width * frame_height
            reletive_face_area = face_area / frame_area
            
            face_id = recognize_face(embeddings, reletive_face_area)

            if face_id is None:
                face_id = update_known_faces(embeddings)
                print(face_id)
            current_ids.add(face_id)

            cv2.rectangle(frame, (int(left), int(top)),
                          (int(right), int(bottom)), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, f"ID: {face_id}, Count: {known_faces[face_id]['count']}",
                        (int(left) + 6, int(bottom) - 6), font, 0.7, (255, 255, 255), 2)

    absent_ids = set(list(known_faces.keys())) - current_ids
    for id in absent_ids:
        known_faces[id]["present"] = False
    current_ids = set()

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
'''
st.code(code, language='python')
