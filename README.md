# Speech Processing Pipeline

This repository contains a Python script for a speech processing pipeline. The pipeline includes recording audio, transcribing the audio, translating the transcription, and serving the translated text through a server.

## Requirements

Ensure you have the following dependencies installed:

- Python 3.x
- Recommended to use a python virtual environment
- Required Python libraries (specified in `requirements.txt`)

## Setup

To install the required dependencies, please follow the instructions below.

1. Clone this repository:

    ```bash
    git clone -b better https://github.com/cclngit/SpeechToTextToTranslate.git
    ```

2. Navigate to the project directory:

    ```bash
    cd SpeechToTextToTranslate
    ```

3. Create a virtual environment:

    ```bash
    python -m venv venv
    ```

4. Activate the virtual environment:

    - On Windows:

        ```bash
        venv\Scripts\activate
        ```

    - On macOS and Linux:

        ```bash
        source venv/bin/activate
        ```

5. Create a model directory and download the speech-to-text and translation models:

    ```bash
    mkdir models
    ct2-transformers-converter --model facebook/nllb-200-distilled-600M --output_dir models/nllb-200-distilled-600M
    ct2-transformers-converter --model openai/whisper-tiny --output_dir models/faster-whisper-tiny --copy_files tokenizer.json --quantization float32
    ```

6. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

7. Configure the pipeline:

    - Modify `config.json` to customize the pipeline settings as needed.

## Usage

To run the speech processing pipeline, execute the `run_pipeline.py` script:

```bash
python src/main.py
```

Make sure to provide a valid configuration file named `config.json` in the repository directory. You can use the provided `config_template.json` as a reference.
But don't forget to download the models as mentioned in the setup section.
Whisper have 4 models :

- whisper-tiny
- whisper-small
- whisper-medium
- whisper-large

And the nllb-200-distilled-600M model can be found on [Huggingface](https://huggingface.co/facebook/nllb-200-distilled-600M) or on [OpenNMT](https://opennmt.net/Models-py/)

## Configuration

The pipeline's behavior can be configured using the `config.json` file. You can customize settings such as directories for recordings, transcriptions, translations, recording frequency, duration, language settings, etc.

## Contributing

Contributions are welcome! If you encounter any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Special thanks to the developers of the libraries and tools used in this project.

## Disclaimer

This project is for educational and demonstration purposes only. Ensure compliance with applicable laws and regulations when using this software in real-world scenarios.