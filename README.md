# ABS audiobook Recommender

A personalized recommendation system for your [Audiobookshelf](https://www.audiobookshelf.org/) library. This tool analyzes your listening history and uses advanced local embeddings to find "hidden gems" in your unread collection. It can optionally use a LLM to generate personalized reasoning for every suggestion.

## Features

-   **Smart Integration**: Connects directly to your Audiobookshelf server to fetch your library and listening progress.
-   **Intelligent Filtering**:
    -   Excludes finished books.
    -   Excludes books currently in progress.
    -   Excludes books marked as **Abandoned**.
    -   **Series Awareness**: Only recommends the *next* unread book in a series or the first book of a new series.
    -   **Abandoned Series Filter**: Automatically excludes entire series if a sequenced book is abandoned.
-   **Advanced Local RAG**:
    -   **Zero-Latency Embeddings**: Uses optimized **Model2Vec** (Jina v3 distilled) for instant, local vectorization.
    -   **Privacy First**: No external APIs or heavy GPU requirements.
    -   **Semantic Discovery**: Finds connections based on plot, themes, and writing style, not just keywords.
-   **Smart Duration Boosting**:
    -   Learns your preferred audiobook lengths (e.g., "Short", "Epic").
    -   Boosts recommendations that match your listening habits while filtering out lengths you avoid.
-   **(Optional) AI-Powered Recommendations**: 
    -   Connect a LLM (like `llama-server`) to get personalized, explained reasons for each recommendation.
    -   If no LLM is provided, the system provides high-quality ranked matches based on similarity scores.
-   **Collaborative Filtering**: Leverages reading patterns from other users on your server to boost relevant recommendations.
-   **Listening History & Ratings**:
    -   View all your finished audiobooks in one place.
    -   Rate books 1-5 stars with an interactive star rating widget.
    -   Books are grouped by series and sorted by sequence number.
    -   Ratings are saved to a local database for future use in recommendations.
-   **Web Interface**: A clean, responsive UI to view recommendations.
-   **Multilingual Support**: Fully localized interface in **English** and **German** with persistent user preferences.
-   **Enhanced UI**:
    -   **Clickable Covers**: Direct links to your Audiobookshelf book pages.
    -   **Expandable Descriptions**: Read full book summaries directly in the card.
    -   **Settings Overlay**: Easy access to language selection and password management.
-   **Privacy Focused**: All analysis happens locally. Book data is only sent to your local LLM (if configured).

## Prerequisites

-   **Python 3.13+**
-   **Node.js 18+ & npm** (Required for building the frontend)
-   **Audiobookshelf Server**
-   **(Optional) LLM Server**: Required only if you want AI-generated explanations. Any OpenAI-compatible server (like `llama.cpp`'s `llama-server`) works.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd ABS_vorschlaege
    ```

2.  **Install dependencies:**
    This project uses [`uv`](https://docs.astral.sh/uv/) for fast dependency management.
    ```bash
    uv sync
    ```

3.  **Build Frontend:**
    Navigate to the frontend directory, install dependencies, and build the static assets:
    ```bash
    cd frontend
    npm install
    npm run build
    cd ..
    ```

4.  **Configure Environment Variables:**
    Create a `.env` file in the root directory:

    ```env
    ABS_URL=http://your-audiobookshelf-url
    ABS_TOKEN=your-audiobookshelf-api-token
    # Optional: Only needed for AI reasons
    LLAMA_SERVER_URL=http://localhost:8080/v1/chat/completions
    # Optional: Limit to specific library ID
    ABS_LIB=
    # Optional: Admin password (default: admin)
    ROOT_PASSWORD=admin
    ```

## Usage

1.  **Start the Web App:**
    ```bash
    uv run web_app/app.py
    ```

2.  **View Recommendations:**
    Open `http://localhost:5000` in your browser.
    
    **Login:** 
    -   **Standard Users**: Log in with your Audiobookshelf username. 
        -   **Initial Password**: Same as your username.
        -   **First Login**: You will be **forced** to change your password immediately upon your first successful login.
    -   **Admin User**: A `root` user is created automatically to manage the application.
        -   **Capabilities**: 
            -   View all registered users.
            -   Reset user passwords.
            -   Force a library synchronization.
        -   **Limitations**: Cannot be used to view recommendations or track listening history.
        -   **Default Login**: `root` / `admin` (change via `ROOT_USERNAME`/`ROOT_PASSWORD` environment variable).
    
    **Settings:** Click the gear icon to change your language or update your password manually.

## Developer Info

### Architecture
-   **RAG System (`web_app/recommend_lib/rag.py`)**:
    -   Automatically downloads and caches the quantized Jina v3 ONNX model (~130MB) for fast, low-memory embedding.
    -   Vector data is persisted in `rag_db_v2/` using two collections (`content` and `metadata`).
-   **Recommendation Logic**:
    -   Unread books are ranked by their similarity to your finished books.
    -   **Weighted Scoring**: Semantic content (60%) and structural metadata (40%) are evaluated separately.
    -   Scores are boosted by:
        -   **User Preferences**: Top authors and genres.
        -   **Collaborative Signals**: High ratings from similar users.
        -   **Duration Matches**: Books fitting the user's preferred length buckets.
    -   Scores are penalized by:
        -   **Negative Signals**: Similarity to abandoned or low-rated books.

See [RECOMMENDATION_ALGO.md](docs/RECOMMENDATION_ALGO.md) for a deep dive into the algorithm.

### Adding New Languages
Want to contribute a translation? See [Adding a New Language](docs/add_new_language.md).

## Roadmap

- [x] Choose what ABS library to use (multiple libraries?)
- [x] Multi-user support
- [x] Login system
- [x] Advanced RAG Integration
- [x] Scoring system for your audiobooks (Listening History with 1-5 star ratings)
- [x] Periodic background updates with caching to get new recommendations automatically after finishing a book without spamming the LLM
- [x] A page for books in progress with the option to mark them as abandoned (also used for recommendations algorithm)
- [ ] Docker containerization for easier deployment
- [x] Enhanced UI/UX design
- [ ] ~~Additional filtering options (e.g., by genre, length, narrator)~~
- [x] Support for other AI models/providers
- [ ] Mobile-friendly design

(Strikethrough items are not actively planned but may be revisited in the future.)


## Background Tasks

The application runs background tasks to keep your recommendations fresh without manual intervention:

1.  **Library Indexing**:
    -   **Frequency**: Every 6 hours (default).
    -   **Action**: Scans your Audiobookshelf library for new or updated books and updates the local RAG index to ensure new books are discoverable.

2.  **Recommendation Updates**:
    -   **Frequency**: Checks every 5 minutes (default).
    -   **Action**: 
        -   If new books were found during indexing, it regenerates recommendations for all users.
        -   If a user has recently rated or finished a book, it regenerates recommendations specific to that user.
        -   Updates are pushed to the web interface in real-time via WebSockets.

### Configuration

You can adjust the frequency of these tasks in `web_app/defaults.py`:

```python
BACKGROUND_TASKS = {
    "CHECK_NEW_BOOKS_INTERVAL": 6, # Hours between library scans
    "CREATE_RECOMMENDATIONS_INTERVAL": 5, # Minutes between activity checks
}
```



## Contributing
Contributions are welcome! Please open issues or submit pull requests (opening an issue with the proposed feature before a PR is encouraged) for bug fixes, features, or improvements.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
