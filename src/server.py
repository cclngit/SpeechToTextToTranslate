from flask import Flask, render_template, request, jsonify
import queue

app = Flask(__name__)

translation_queue = queue.Queue()

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/stream')
def stream():
    def generate():
        while True:
            translation = translation_queue.get()
            yield f"data: {translation}\n\n"

    return app.response_class(generate(), mimetype='text/event-stream')

@app.route('/send_translation', methods=['POST'])
def send_translation():
    translation = request.form['translation']
    translation_queue.put(translation)
    return 'OK'

def run_server(queue):
    global translation_queue
    translation_queue = queue
    app.run(debug=True, use_reloader=False)

if __name__ == '__main__':
    try:
        run_server(queue.Queue())
    except KeyboardInterrupt:
        print('Server stopped')
        exit(0)

