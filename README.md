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
-   **User Authentication**: Secure login system to access personalized recommendations.

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
    LANGUAGE=<desired-language-code>
    ABS_LIB=<library-id>
    ```

    *   **ABS_URL**: The full URL to your Audiobookshelf server (e.g., `http://192.168.1.100:13378`).
    *   **ABS_TOKEN**: The root API token for your Audiobookshelf server (found in Settings > Users > Root User).
    *   **GEMINI_API_KEY**: Your API key from Google AI Studio.
    *   **LANGUAGE**: (Optional) The language code for recommendations (e.g., `de` for German, `en` for English).You can add your own translations in `web_app/recommend_lib/languages` folder and use the filename as the language code.
    *   **ABS_LIB**: (Optional) The ID of the library you want to use. If not set, all libraries will be used. This is useful if you want to restrict recommendations to a specific library (e.g. Audiobooks). In this way you can still get the audiobook version recommended even if you have finished the ebook version.

## Usage

1.  **Start the Web App:**
    ```bash
    python web_app/app.py
    ```

2.  **Log In & View Recommendations:**
    Open your browser and go to `http://localhost:5000`.

    **Login Credentials:**
    *   **Username**: Your Audiobookshelf username.
    *   **Password**: Your Audiobookshelf username (case-sensitive).

    *Note: The app syncs users from your ABS instance. By default, the password is set to be the same as the username.*

## Project Structure

```
ABS_vorschlaege/
├── .env                    # Your environment variables (not in git)
├── pyproject.toml          # Project metadata and dependencies
├── README.md
└── web_app/
    ├── app.py              # Flask web server entry point
    ├── db.py               # Database connection and models
    ├── recommend_lib/      # Core recommendation logic
    │   ├── abs_api.py      # Audiobookshelf API client
    │   ├── gemini.py       # Google Gemini API integration
    │   └── recommender.py  # Main recommendation orchestration
    ├── static/             # CSS and other static assets
    └── templates/          # HTML templates
```

## Roadmap

- [x] Support for more languages (currently the prompt is in German only)
- [x] Choose what ABS library to use (multiple libraries?)
- [x] Multi-user support
- [x] Login system
- [ ] Periodic background updates with caching to get new recommendations automatically after finishing a book without spamming the Gemini API
- [ ] Docker containerization for easier deployment
- [ ] Enhanced UI/UX design
- [ ] ~~Additional filtering options (e.g., by genre, length, narrator)~~
- [ ] ~~Support for other AI models/providers~~ 
- [ ] Mobile-friendly design

(Strikethrough items are not actively planned but may be revisited in the future.)


## Contributing
Contributions are welcome! Please open issues or submit pull requests (opening an issue with the proposed feature before a PR is encouraged) for bug fixes, features, or improvements.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

 

