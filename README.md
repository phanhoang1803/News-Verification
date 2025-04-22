# DKH-Thesis

## Installation

1. Clone the repository
```bash
git clone https://github.com/phanhoang1803/News-Verification.git
cd News-Verification
```

2. Set up a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
.\venv\Scripts\activate   # On Windows
```

3. Install dependencies and project
```bash
pip install -r requirements.txt
```

5. Set up environment variables by creating a .env file in the root directory. You can use the .env.template file as a template.

## Usage

### Dataset creation

Please refer to the [Dataset Creation](https://github.com/phanhoang1803/News-Verification/tree/main/src/dataset_creation) for more details.


### Inference

To run the inference on news clipping dataset:

```bash
python .\src\inference_newsclippings.py \
--data_path `path/to/test_dataset` \
--entities_path `path/to/links_test.json` \
--image_evidences_path `path/to/inverse_search/test/test.json` \
--text_evidences_path `path/to/direct_search/test/test.json` \
--context_dir_path `path/to/context/test` \
--gemini_api_key `your_gemini_api_key` \
--skip_existing \
--start_idx `start_index` \
--end_idx `end_index`
```

### Output structure

The output will be saved in the `result` folder.

```bash
result/
├── <id>.json
├── <id>.json
├── ...
```


