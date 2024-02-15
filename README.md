# Overview

This project aims to extract ESG information such as climate-related risks and climate-related opportunities from annual sustainability reports using the Large Language Model, ClimateBERT. The pipeline consists of several downstream tasks including Climate-Detector, TCFD-Domain, and Climate-Sentiment.

# Requirements

- Python 3.x
- PyTorch
- Other dependencies listed in `requirements.txt`

# Installation

1. Clone this repository.

2. Install the required dependencies by running:

    `$ pip install -r requirements.txt`


# Usage

1. Prepare your annual sustainability reports in PDF format. Store the PDFs here: `./pdfs/`
2. Run the pipeline script.

    `$ python main.py`


# Fine-Tuning a Custom Downstream Task

To fine-tune ClimateBERT for a custom downstream task, follow these steps:

1. **Data Preparation**: Prepare your dataset for the downstream task, ensuring it is labeled appropriately.

2. **Fine-Tuning Script**: Modify the fine-tuning script provided in the repository according to your task requirements.

3. **Model Configuration**: Adjust the model configuration and hyperparameters as needed for your task.

4. **Training**: Train the fine-tuned model using the prepared dataset. Monitor performance metrics and adjust as necessary.

5. **Evaluation**: Evaluate the fine-tuned model on a separate validation set to assess its performance.

6. **Deployment**: Once satisfied with the model's performance, deploy it for inference on new data.


For detailed instructions on fine-tuning ClimateBERT, refer to the [paper](https://arxiv.org/abs/2110.12010) and the official documentation on 🤗 [Hugging Face](https://huggingface.co/climatebert).

# Acknowledgements

This project utilizes ClimateBERT, developed by [authors of the paper](https://arxiv.org/abs/2110.12010).
