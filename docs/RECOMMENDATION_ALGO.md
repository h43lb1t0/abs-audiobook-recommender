# Recommendation Algorithm Documentation

This document explains the inner workings of the audiobook recommendation system used in this project. The system employs a hybrid approach combining **Retrieval-Augmented Generation (RAG)**, **Collaborative Filtering**, **User Preference Weighting**, and optionally an **LLM (Large Language Model)** for final presentation.

## Overview

The recommendation engine is designed to suggest books based on a user's reading history, specific ratings, and implied taste profiles. It operates in several distinct stages:

1.  **Data Ingestion & Filtering**: Fetching books and filtering out finished/in-progress items, including **Abandoned** books.
2.  **RAG-Based Scoring**: Using vector embeddings to find books similar to what the user likes.
3.  **User Preference Boosting**: Weighting candidates based on top genres and authors.
4.  **Collaborative Filtering**: Boosting books liked by similar users.
5.  **Duration Boosting**: Adjusting scores based on the user's preferred audiobook lengths.
6.  **LLM Refinement (Optional)**: Using an LLM to select and explain the final recommendations.

---

## 1. Data Ingestion & Filtering

Before any ranking occurs, the system compiles a list of valid candidate books.

-   **Source**: Fetches all items from the Audiobookshelf library.
-   **Exclusions**:
    -   Books already marked as `finished`.
    -   Books currently `in_progress`.
    -   Books marked as `abandoned` by the user.
    -   Books that duplicate the `(title, author)` of finished/abandoned books.
-   **Series Logic**:
    -   If a book is part of a series, the system ensures *sequential consistency*.
    -   It only recommends the **next** unread book in the series sequence (e.g., Book 2 is only a candidate if Book 1 is finished).
    -   **Abandoned Series Filter**: If any book in a sequenced series is marked as `abandoned`, the **entire series** is removed from candidates.
    -   If no sequence number is available, the first available book in the series is chosen.

## 2. RAG-Based Scoring (The Core)

The core mechanism uses **Vector Embeddings** to understand similarity between books. The system uses a **Two-Stream** approach to separate "what a book is about" from "who made it."

### The RAG System
The system maintains two distinct vector collections in **ChromaDB**:

1.  **Content Collection** (`audiobooks_content_v1`):
    -   **Purpose**: Semantic matching of plot, mood, and themes.
    -   **Input**: Genres + Tags + Description.
    -   **Weight**: **60%** of the matching score.

2.  **Metadata Collection** (`audiobooks_metadata_v1`):
    -   **Purpose**: Structural matching of creators and format.
    -   **Input**: Title + Author + Narrator + Series.
    -   **Weight**: **40%** of the matching score.

### Query & Penalty Phases

The system calculates scores by comparing the user's history against the candidates using a two-phase approach:

1.  **Positive Phase (Query)**:
    -   **Input**: Books rated **4-5 Stars**. (Fallback: All finished books if no ratings exist).
    -   **Action**: Finds similar unread books.
    -   **Weight Strategy**: Matches from positive queries receive a **2.0x** weight multiplier. (Fallback queries use 1.0x).

2.  **Negative Phase (Penalty)**:
    -   **Input**: Books rated **1-2 Stars** AND books marked as **Abandoned**.
    -   **Action**: Finds similar unread books (Context & Metadata).
    -   **Penalty**: If a candidate is similar to a negative/abandoned book, its score is **reduced**.
    -   **Weight Strategy**: Matches from negative queries carry a **-1.5x** penalty multiplier.

### Ranking Algorithm
The system ranks unread books using a **Normalized Weighted Average** approach.

#### 1. Score Calculation
The system calculates two raw scores for every candidate:
1.  **RAG Score (Similarity)**: Derived from the cosine distance between User and Book embeddings (Content + Metadata).
2.  **Preference Score (Explicit)**: Derived from explicit matches with the user's top Genres, Authors, and Narrators.

#### 2. Normalization & Weighting
Each booklet's score is normalized against the batch maximums:
-   `Norm_RAG = Raw_RAG / Max_RAG`
-   `Norm_Pref = Raw_Pref / Max_Pref`

The final base score is a weighted combination:
`Final_Score = (Norm_RAG * 0.7) + (Norm_Pref * 0.3)`

> This 70/30 split prioritizes semantic relevance (Plot/Theme) while still using User Preferences (Author/Genre) as a significant tie-breaker.

## 3. User Preference Boosting

On top of semantic similarity, the system calculates a raw preference score which is then normalized:
-   **Top Genres**: +10 (raw)
-   **Top Authors**: +15 (raw)
-   **Top Narrators**: +5 (raw)

## 4. Collaborative Filtering

The system attempts to find "reading soulmates" to diversify recommendations.

1.  **User Similarity**: It compares the current user's *taste clusters* with every other user's *finished book embeddings*.
2.  **Matching**: If another user has a taste cluster that is highly similar (cosine similarity > **0.5**).
3.  **Boosting**:
    -   Books liked by the similar user (from their "Positive" list) receive a boost.
    -   **Scaled Boost**: `Score += 0.15 * Similarity` (Scaled to match the 0-1 normalized system).
    -   **Match Reason**: These books are flagged with "Highly relevant to similar user 'Username'".

## 5. Duration Boosting (Smart Post-Processing)

To account for user preferences regarding audiobook length, the system employs a boosting mechanism that corrects for **Availability Bias**.

### 1. Buckets
Books are categorized into duration buckets:
- **Super Short**: < 1 hour
- **Short**: 1 - 3 hours
- **Mid-Short**: 3 - 5 hours
- **Medium**: 5 - 15 hours
- **Long**: 15 - 24 hours
- **Epic**: > 24 hours

### 2. Neighbor Bleed (Soft Preferences)
User preferences "bleed" into neighbor buckets to smooth out the data.
- **Formula**: `SmoothedAffinity[i] = RawAffinity[i] + (RawAffinity[Neighbors] * 0.1)` (10% bleed).

### 3. Availability Bias Correction (Lift)
We calculate **Lift** to find true preferences relative to the library's supply.
- **Formula**: `Lift = min( (UserSmoothedShare / LibraryShare), 2.5 )`

### 4. Strictness Gating & Sigmoid Scoring
The system applies a **Sigmoid Multiplier** to the final score.

1.  **Strictness Gating**:
    -   If a user's affinity for a bucket is below the threshold (< **2%** share), the system applies a **Strictness Penalty**.
    -   **Multiplier**: `0.1x`.

2.  **Sigmoid Multiplier**:
    -   If the user meets the strictness threshold, a logistic curve is applied based on Lift.
    -   **Formula**: `Multiplier = 2.0 / (1 + e^(-k * (Lift - 1)))` (where **k=1.5**).
    -   **Result**:
        -   *Lift = 1.0 (Neutral)*: Multiplier = `1.0`.
        -   *Lift < 1.0 (Low Preference)*: Multiplier approaches `0.3-0.4`.
        -   *Lift > 1.0 (High Preference)*: Multiplier approaches `2.0`.

### 5. Application
The final score is multiplied by this duration factor:
`FinalScore = Score * DurationMultiplier`

## 6. Final Selection & LLM Integration

### Candidate Selection
The top **50** books with the highest calculated scores are selected as candidates.

### Mode A: RAG-Only (Fast)
If the LLM is disabled:
-   Returns the top 20 candidates directly.
-   Generates static reasons based on score breakdown (e.g., "Matches your reading profile", "Similar to books you loved", "Strong Content & Style Match").

### Mode B: LLM Generation (Smart)
If the LLM is enabled:
1.  **Prompt Construction**: A text prompt is built containing:
    -   A list of the User's recently finished books.
    -   The list of Top 50 Candidates (Title, Author, Series, Description).
    -   A system instruction (loaded from `languages/user/[lang].txt`) telling the LLM to pick the best fits and explain why.
    -   **Note**: The system separates logic from prompts using external text files.
2.  **Generation**: The LLM (e.g., local Llama via Ollama or OpenAI compatible API) returns a JSON response with the selected books and personalized reasons.
3.  **Mapping**: The system maps the LLM's selected IDs back to the original book objects.

## Architecture Diagram

```mermaid
graph TD
    A[All Books] --> B{Filter}
    B -->|Remove Finished/InProgress/Abandoned| C[Candidates]
    B -->|Series Logic| C
    B -->|Abandoned Series| X[Discard]
    
    D[User History] --> E{Ratings?}
    E -->|4-5 Stars| F[Positive Seeds (+2.0x)]
    E -->|1-2 Stars| G[Negative Seeds (-1.5x)]
    E -->|Abandoned| G
    E -->|Fallback| H[All Finished Seeds (+1.0x)]
    
    F --> I[K-Means Clustering]
    H --> I
    
    I --> J[Cluster Centers]
    J -->|Query RAG| K[Similarity Scores]
    
    G -->|Query RAG| L[Penalty Scores]
    
    C --> M[Score Calculation]
    K --> M
    L --> M
    
    N[User Top Stats] -->|Genre/Author/Narrator Boost| M

    O[Other Users] -->|Collaborative Filter >0.5| M
    
    M --> P[Top 50 Ranked]
    
    P --> Z{Duration Boost}
    Z -->|Sigmoid Multiplier| P_Final[Final Ranked List]
    
    P_Final --> Q{Use LLM?}
    Q -->|No| R[Top 20 Dictionary]
    Q -->|Yes| S[LLM Selection & Reasoning]
    S --> T[Final Recommendations]
```
