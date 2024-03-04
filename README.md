# Speech Processing Pipeline

This repository contains a Python script for a speech processing pipeline. The pipeline includes recording audio, transcribing the audio, translating the transcription, and serving the translated text through a server.

## Installation

To install the required dependencies, please follow the instructions below.

1. Clone the repository:

```bash
git clone -b main https://github.com/cclngit/SpeechToTextToTranslate.git
cd SpeechToTextToTranslate
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Create a directory for the models and run convert_file.sh:

```bash
mkdir models
bash convert_file.sh
```

## Usage

To run the speech processing pipeline, execute the `run_pipeline.py` script:

```bash
python src/main.py
```

Make sure to provide a valid configuration file named `config.json` in the repository directory. You can use the provided `config_template.json` as a reference.

## Configuration

The pipeline's behavior can be configured using the `config.json` file. You can customize settings such as directories for recordings, transcriptions, translations, recording frequency, duration, language settings, etc.

## Contributing

Contributions are welcome! If you encounter any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## requirements.txt

```txt
numpy
torch
transformers
pyaudio
sounddevice
wavio
flask
librosa
transformers
ctranslate2
--extra-index-url https://download.pytorch.org/whl/cu116
git+https://github.com/openai/whisper.git
```