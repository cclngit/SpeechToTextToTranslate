# Speech Translation Pipeline

This Python script demonstrates a speech translation pipeline. It transcribes spoken input, translates it to a target language, and serves the translated text. The pipeline utilizes multithreading to handle these tasks concurrently.

## Requirements

Ensure you have the following dependencies installed:

- Python 3.x
- Recommended to use a python virtual environment
- Required Python libraries (specified in `requirements.txt`)

## Setup

1. Clone this repository:

    ```bash
    git clone -b better https://github.com/cclngit/SpeechToTextToTranslate.git
    ```

2. Navigate to the project directory:

    ```bash
    cd speech-translation-pipeline
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

5. Create a model directory and download the translation model using the `ct2-transformers-converter` tool. For example, to download the `facebook/nllb-200-distilled-600M` model, run the following command:

    ```bash
    mkdir models
    ct2-transformers-converter --model facebook/nllb-200-distilled-600M --output_dir models/nllb-200-distilled-600M
    ```

6. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

7. Configure the pipeline:

    - Modify `config.json` to customize the pipeline settings as needed.

## Usage

Run the pipeline script:

```bash
python pipeline.py
```

## Configuration

The `config.json` file contains the configuration settings for the pipeline. Adjust the following parameters according to your requirements:

- **Transcribe Configuration:**
  - `model`: Speech recognition model to use.
  - `energy_threshold`: Energy threshold for audio input.
  - `record_timeout`: Maximum duration for recording audio.
  - `phrase_timeout`: Timeout for detecting phrases in audio.

- **Translate Configuration:**
  - `translator`: Translation service/API to use.
  - `tokenizer`: Tokenization method for text.
  - `device`: Device for translation (e.g., CPU, GPU).
  - `src_lang`: Source language for translation.
  - `tgt_lang`: Target language for translation.

## Contributing

Feel free to contribute to this project by creating issues or pull requests. Contributions are welcome!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Special thanks to the developers of the libraries and tools used in this project.
- One portion of this code is based on the work of [davabase](https://github.com/davabase/whisper_real_time).

## Disclaimer

This project is for educational and demonstration purposes only. Ensure compliance with applicable laws and regulations when using this software in real-world scenarios.