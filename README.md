# Gemma 3: Local Deployment and Inference

This repository provides a step-by-step demonstration on deploying the latest Gemma model (Gemma 3) on your own server. By following the guide, you can easily perform inference entirely within your own system, ensuring your data remains secure and never leaves your environment.

## Features

* Deploy the cutting-edge Gemma 3 model locally
* Secure inference without external data transfer
* Easy-to-follow deployment instructions

## Prerequisites

* Server or virtual machine with sufficient resources (CPU/GPU and RAM)
* Python environment (recommended Python 3.9 or newer)
* Docker (optional but recommended for simplified deployment)

## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/ardyadipta/gde_gemma_text2sql.git
   cd gde_gemma_text2sql
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download Gemma 3 Model**
   Instructions and links to download the model weights from the official source.

4. **Configure Environment Variables**
   Set required configurations in `.env` file (Kaggle username and password to download model).

## Running the Deployment

1. Run the Gemma server in the directory gemma-web-service from your server
2. from your own local system or laptop, run chatbot.py as a simple chatbot project talk to Gemma deployed on your server

## Performing Inference

After successful deployment, send requests to your local server endpoint:

```bash
curl -X POST http://localhost:8000/infer -d '{"input": "Your inference prompt"}'
```

## Security

By hosting Gemma 3 locally, you ensure maximum privacy and security. No data used during inference will leave your infrastructure.

## Contributions

Feel free to open an issue or submit a pull request to enhance this project.

## License

This repository is licensed under the MIT License. See [LICENSE](LICENSE) for details.
