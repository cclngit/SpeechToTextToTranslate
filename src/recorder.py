import sounddevice as sd
import wavio as wv
import datetime
import os, glob

freq = 44100
duration = 5 # in seconds
recordings_dir = os.path.join('./recordings', '*')

print('Recording')

try:
    while True:
        recording = sd.rec(int(duration * freq), samplerate=freq, channels=2)
        sd.wait()
        filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".wav"
        wv.write(f"./recordings/{filename}", recording, freq, sampwidth=2)
        print(f"Recording saved as {filename}")
        
        # delete everything in the recordings directory except for the 10 most recent recording
        files = sorted(glob.iglob(recordings_dir), key=os.path.getctime, reverse=True)
        for i in range(10, len(files)):
            os.remove(files[i])
        print("Old recordings deleted")
        
        
except KeyboardInterrupt:
    print('Recording stopped')
    exit(0)

    


    
