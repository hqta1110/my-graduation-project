# 🌿 Plant AI Chatbot

## Project Structure

This project is organized into distinct folders to separate the different components of the application. The full source code is structured as follows:

  * `📁 app/`
    This folder contains all the backend code. It includes the Python files for the FastAPI server, the AI models' logic, and database interactions.

  * `📁 data/`
    This holds the necessary data for the application, such as the plant metadata `.json` files and the pre-computed FAISS index. For the final application, this data is downloaded from Hugging Face upon startup.

  * `📁 frontend/`
    This contains all the source code for the user interface (UI) that is displayed in the web browser.

  * `📁 models/`
    This folder stores the trained AI model checkpoints (`.pth` files). Like the data, these are downloaded from Hugging Face when the application starts.

  * `📁 startup/`
    This directory holds the files needed for easy deployment by end-users. It contains the `docker-compose.yml` file and the `sample.env.txt` template.

## How to Run the Application

These instructions are for end-users who have been given the `startup` folder.

### Prerequisites

You must have **Docker** installed.

-----

### 1\. Prepare Files

Place the `docker-compose.yml` and `sample.env` files into a **new, empty** directory on your machine.

-----

### 2\. Set Up API Keys

You need to provide API keys for the application to work.

1.  Create a copy of `sample.env` and rename it to `.env`.

    ```bash
    cp sample.env .env
    ```

2.  Open the `.env` file and fill in your secret keys for the following variables:

      * `HUGGINGFACE_TOKEN`: Your access token from Hugging Face.
      * `GEMINI_API_KEY`: Your API key for the Google Gemini model.

    Your `.env` file should look like this:

    ```env
    HUGGINGFACE_TOKEN="hf_your_huggingface_token_here"
    GEMINI_API_KEY="AIza_your_google_api_key_here"
    ```

-----

### 3\. Launch the App

Open a terminal in the directory containing your files and run the following command:

```bash
docker compose up
```

This command will download the necessary Docker images and start the application. The first launch may take 10-15 minutes for downloading checkpoints from the Huggingface Hub.

-----

### 4\. Access the Application

Once the containers are running, open your web browser and go to:

**http://localhost:9697**

-----

### 5\. Stop the Application

To stop the running application, execute this command in the same terminal:

```bash
docker compose down
```

-----

### ⚠️ Note on Image Data

For easier sharing and a quicker startup, the current version only includes **one representative image per species**. The main features like identification and Q\&A will still function normally.

However, to use the full **Image Library** feature and view all images for each species, you need to download and place the complete image database into `app/plant_data` directory.
