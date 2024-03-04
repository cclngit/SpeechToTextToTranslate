import sounddevice as sd
import wavio as wv
import datetime
import queue
import utils


def record_and_save(queue1, recordings_dir, freq, duration):
    try:
        while True:
            recording = sd.rec(int(duration * freq), samplerate=freq, channels=2)
            sd.wait()
            filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".wav"
            wv.write(f"{recordings_dir}/{filename}", recording, freq, sampwidth=2)
            print(f"Recording saved as {filename}")
            queue1.put(f"{recordings_dir}/{filename}")
            utils.delete_old_files(f"{recordings_dir}/*.wav")
    
    except Exception as e:
        print(f"Error recording: {e}")

if __name__ == "__main__":
    queue1 = queue.Queue()
    recordings_dir = "recordings"
    freq = 44100
    duration = 10
    record_and_save(queue1, recordings_dir, freq, duration)


