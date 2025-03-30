# proxy-OAI-compat

## Description
This repository hosts a high-performance proxy server designed to be fully compatible with the OAI-compat protocol. Developed using Python and powered by the FastAPI framework, it provides an efficient, scalable, and user-friendly solution for handling requests.

> **Note:** This project currently supports **MacOS with M-series chips** only.

## Installation

Follow these steps to set up the proxy server:

### Native Python Recommendation
We recommend using a native Python version (supporting ARM architecture). The development environment for this project uses Python `3.11`.

### Setup Steps
1. Create a virtual environment for the project:
    ```bash
    python3 -m venv oai-compat-server
    ```

2. Activate the virtual environment:
    ```bash
    source oai-compat-server/bin/activate
    ```

3. Install the required Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Troubleshooting
**Issue:** My OS and Python versions meet the requirements, but `pip` cannot find a matching distribution.

**Cause:** You might be using a non-native Python version. Run the following command to check:
```bash
python -c "import platform; print(platform.processor())"
```
If the output is `i386` (on an M-series machine), you are using a non-native Python. Switch to a native Python version. A good approach is to use [Conda](https://stackoverflow.com/questions/65415996/how-to-specify-the-architecture-or-platform-for-a-new-conda-environment-apple).


## Usage
To start the proxy server, activate the virtual environment and run the main application file:
```bash
source oai-compat-server/bin/activate
python app.main --model-path <path-to-model> --model-type <model-type>
```

The server will start and listen for incoming requests. You can configure server settings in the `config.json` file.

## API Usage

### Vision Request Example
You can make vision requests to analyze images using the `/v1/chat/completions` endpoint. Here's an example:

```bash
curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2-VL-2B-Instruct-4bit",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What is in this image?"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
            }
          }
        ]
      }
    ] 
  }'
```

### An example of a vision request
- `messages`: Array of message objects containing:
  - `role`: The role of the message sender ("user" or "assistant")
  - `content`: For vision requests, an array of content objects:
    - `type`: Either "text" or "image_url"
    - `text`: The text prompt (for type "text")
    - `image_url`: Object containing the image URL (for type "image_url")

### Response Format
The server will return responses in OpenAI-compatible format:

```json
{
  "id": "chatcmpl-1234567890",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "mlx-community/Qwen2-VL-2B-Instruct-4bit",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The image shows a wooden boardwalk..."
      },
      "finish_reason": "stop"
    }
  ]
}
```

### Streaming Responses
For streaming responses, add `"stream": true` to your request. The response will be in Server-Sent Events (SSE) format:

```bash
curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "stream": true,
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What is in this image?"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://example.com/image.jpg"
            }
          }
        ]
      }
    ] 
  }'
```

## Contributing
We welcome contributions to improve this project! Here's how you can contribute:
1. Fork the repository to your GitHub account.
2. Create a new branch for your feature or bug fix:
    ```bash
    git checkout -b feature-name
    ```
3. Commit your changes with clear and concise messages:
    ```bash
    git commit -m "Add feature-name"
    ```
4. Push your branch to your forked repository:
    ```bash
    git push origin feature-name
    ```
5. Open a pull request to the main repository for review.

## License
This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute it under the terms of the license.

## Support
If you encounter any issues or have questions, please:
- Open an issue in the repository.
- Contact the maintainers via the provided contact information.

Stay tuned for updates and enhancements!

## Acknowledgments
Special thanks to the contributors and the open-source community for their support and inspiration.
