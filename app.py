from flask import Flask, render_template, Response
from camera import Video
from mask import MaskVideo
from gender import Gender

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('base.html')


@app.route('/emotion/')
def emotion():
    return render_template('index.html')


@app.route('/mask/')
def mask():
    return render_template('mask.html')


@app.route('/gender/')
def gender():
    return render_template('gender.html')


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'b'Content-Type:  image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video')
def video():
    return Response(gen(Video()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/maskVideo')
def maskVideo():
    return Response(gen(MaskVideo()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/genderVideo')
def genderVideo():
    return Response(gen(Gender()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


app.run(debug=True)
