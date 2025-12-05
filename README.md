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
    ABS_FETCH_INTERVAL=5
    USE_GEMINI=True
    ```

    *   **ABS_URL**: The full URL to your Audiobookshelf server (e.g., `http://192.168.1.100:13378`).
    *   **ABS_TOKEN**: The root API token for your Audiobookshelf server (found in Settings > Users > Root User).
    *   **GEMINI_API_KEY**: Your API key from Google AI Studio.
    *   **LANGUAGE**: (Optional) The language code for recommendations (e.g., `de` for German, `en` for English).You can add your own translations in `web_app/recommend_lib/languages` folder and use the filename as the language code.
    *   **ABS_LIB**: (Optional) The ID of the library you want to use. If not set, all libraries will be used. This is useful if you want to restrict recommendations to a specific library (e.g. Audiobooks). In this way you can still get the audiobook version recommended even if you have finished the ebook version.
    *   **ABS_FETCH_INTERVAL**: (Optional) The interval in minutes for the background task to check for new finished books and generate recommendations (default: `5`).
    *   **USE_GEMINI**: (Optional) Set to `True` to enable AI recommendations. If `False` or unset, the system will use mock data for testing purposes (default: `False`).

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

3.  **Background Updates:**
    The application automatically checks for new finished books in the background (default every 5 minutes). 
    *   If a new finished book is found, it generates new recommendations.
    *   If you are on the website, the new recommendations will appear automatically via real-time updates.


## Project Structure

```
ABS_vorschlaege/
├── .env                    # Your environment variables (not in git)
├── pyproject.toml          # Project metadata and dependencies
├── uv.lock                 # Lock file for dependencies
├── README.md
└── web_app/
    ├── app.py              # Flask web server entry point
    ├── models/             # Database models
    │   └── db.py           # Database connection and schema
    ├── recommend_lib/      # Core recommendation logic
    │   ├── abs_api.py      # Audiobookshelf API client
    │   ├── gemini.py       # Google Gemini API integration
    │   ├── recommender.py  # Main recommendation orchestration
    │   └── languages/      # Language prompt templates
    ├── static/             # CSS and other static assets
    └── templates/          # HTML templates
```

## Roadmap

- [x] Support for more languages (currently the prompt is in German only)
- [x] Choose what ABS library to use (multiple libraries?)
- [x] Multi-user support
- [x] Login system
- [x] Periodic background updates with caching to get new recommendations automatically after finishing a book without spamming the Gemini API
- [ ] Docker containerization for easier deployment
- [ ] Enhanced UI/UX design
- [ ] ~~Additional filtering options (e.g., by genre, length, narrator)~~
- [ ] ~~Support for other AI models/providers~~ 
- [ ] Mobile-friendly design

(Strikethrough items are not actively planned but may be revisited in the future.)


## Development & Testing

### Mock Recommendations

For development purposes, you can disable the Gemini API to avoid using your quota. Set `USE_GEMINI=False` in your `.env` file.

When enabled, the system generates mock recommendations:
-   **Randomized Titles**: Mock book titles include a random number (e.g., "Project Hail Mary 42") to verify that the UI correctly updates when new recommendations are generated.
-   **Persistence**: Mocks are saved to the database just like real recommendations.
-   **Debugging**: Useful for testing the background scheduler and WebSocket real-time updates.

## Contributing
Contributions are welcome! Please open issues or submit pull requests (opening an issue with the proposed feature before a PR is encouraged) for bug fixes, features, or improvements.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

 

