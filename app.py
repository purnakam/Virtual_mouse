from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import cv2
import HandTrackingModule as htm

app = Flask(__name__)
socketio = SocketIO(app)

detector = htm.HandDetector(max_num_hands=1)
cap = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            img = detector.findHands(frame)
            lmList, bbox = detector.findPosition(img)

            if lmList:
                x1, y1 = lmList[8][1:3]
                x2, y2 = lmList[12][1:3]
                fingers = detector.fingersUp()

                socketio.emit('gesture_data', {'x': x1, 'y': y1, 'fingers': fingers})

            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('frame')
def handle_frame(data):
    x, y = data['x'], data['y']
    emit('response', {'action': 'move', 'x': x + 50, 'y': y + 50})

@socketio.on('webcam_status')
def handle_webcam_status(data):
    if data['status'] == 'started':
        print("Webcam has started.")
    elif data['status'] == 'stopped':
        print("Webcam has stopped.")

if __name__ == '__main__':
    try:
        socketio.run(app, debug=True)
    finally:
        cap.release()
