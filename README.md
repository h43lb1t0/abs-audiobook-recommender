# ABS audiobook Recommender (Audiobookshelf audiobook Recommendations)

A personalized recommendation system for your [Audiobookshelf](https://www.audiobookshelf.org/) library. This tool analyzes your listening history and uses Google's Gemini AI to suggest the next audiobook you should listen to from your unread collection.

## Features

-   **Smart Integration**: Connects directly to your Audiobookshelf server to fetch your library and listening progress.
-   **Intelligent Filtering**:
    -   Excludes books you've already finished.
    -   Excludes books currently in progress.
    -   **Series Awareness**: Only recommends the *next* unread book in a series or the first book of a new series.
-   **AI-Powered Recommendations**: Uses Google Gemini (model `gemini-2.5-pro`, with a free API key you get 50 requests per day. Last checked: 04.12.2025) to analyze your taste profile based on finished books and suggests hidden gems from your library.
-   **Web Interface**: A simple, clean web interface to view recommendations with cover art and AI-generated reasons.
-   **Privacy Focused**: Only sends book titles and authors to Gemini, not your full library data or personal info.

## Prerequisites

-   **Python 3.13+**
-   **Audiobookshelf Server**: You need a running instance of Audiobookshelf.
-   **Google Gemini API Key**: You can get one for free from [Google AI Studio](https://aistudio.google.com/).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd ABS_vorschlaege
    ```

2.  **Install dependencies:**
    This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management.
    ```bash
    uv sync
    ```
    
    *Alternatively, you can install dependencies with pip:*
    ```bash
    pip install flask google-genai python-dotenv requests
    ```

3.  **Configure Environment Variables:**
    Create a `.env` file in the root directory and add the following:

    ```env
    ABS_URL=http://your-audiobookshelf-url
    ABS_TOKEN=your-audiobookshelf-api-token
    GEMINI_API_KEY=your-gemini-api-key
    ```

    *   **ABS_URL**: The full URL to your Audiobookshelf server (e.g., `http://192.168.1.100:13378`).
    *   **ABS_TOKEN**: Generate a token in your ABS user settings.
    *   **GEMINI_API_KEY**: Your API key from Google AI Studio.

## Usage

1.  **Start the Web App:**
    ```bash
    python web_app/app.py
    ```

2.  **View Recommendations:**
    Open your browser and go to `http://localhost:5000`.
    The app will fetch your data, generate recommendations, and display them.

## Project Structure

```
ABS_vorschlaege/
├── .env                    # Your environment variables (not in git)
├── pyproject.toml          # Project metadata and dependencies
├── README.md
└── web_app/
    ├── app.py              # Flask web server entry point
    ├── recommend_lib/      # Core recommendation logic
    │   ├── abs_api.py      # Audiobookshelf API client
    │   ├── gemini.py       # Google Gemini API integration
    │   └── recommender.py  # Main recommendation orchestration
    ├── static/             # CSS and other static assets
    └── templates/          # HTML templates
```

## Roadmap

- [ ] Support for more languages (currently the prompt is in German only)
- [ ] Multi-user support
- [ ] Login system
- [ ] Periodic background updates with caching to get new recommendations automatically after finishing a book without spamming the Gemini API
- [ ] Docker containerization for easier deployment
- [ ] Enhanced UI/UX design
- [ ] ~~Additional filtering options (e.g., by genre, length, narrator)~~
- [ ] ~~Support for other AI models/providers~~ 
- [ ] Mobile-friendly design

(Strikethrough items are not actively planned but may be revisited in the future.)

 

