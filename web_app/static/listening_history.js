document.addEventListener('DOMContentLoaded', () => {
    const resultsContainer = document.getElementById('results');
    const errorMessage = document.getElementById('error-message');
    const loadingContainer = document.getElementById('loading');
    const statsBar = document.getElementById('stats');
    const bookCountEl = document.getElementById('book-count');
    const ratedCountEl = document.getElementById('rated-count');

    // Load listening history on page load
    loadListeningHistory();

    async function loadListeningHistory() {
        try {
            const response = await fetch('/api/listening-history');

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || 'Failed to fetch listening history');
            }

            const books = await response.json();

            if (books.length === 0) {
                loadingContainer.classList.add('hidden');
                errorMessage.textContent = 'No finished audiobooks found. Start listening!';
                errorMessage.classList.remove('hidden');
                return;
            }

            renderBooks(books);
            updateStats(books);

            loadingContainer.classList.add('hidden');
            statsBar.classList.remove('hidden');
            resultsContainer.classList.remove('hidden');

        } catch (error) {
            loadingContainer.classList.add('hidden');
            errorMessage.textContent = error.message;
            errorMessage.classList.remove('hidden');
        }
    }

    function updateStats(books) {
        const ratedCount = books.filter(b => b.rating !== null && b.rating !== undefined).length;
        bookCountEl.textContent = `${books.length} book${books.length !== 1 ? 's' : ''}`;
        ratedCountEl.textContent = `${ratedCount} rated`;
    }

    function renderBooks(books) {
        resultsContainer.innerHTML = '';

        // Group books by series
        const seriesGroups = {};
        const standaloneBooks = [];

        books.forEach(book => {
            if (book.series) {
                if (!seriesGroups[book.series]) {
                    seriesGroups[book.series] = [];
                }
                seriesGroups[book.series].push(book);
            } else {
                standaloneBooks.push(book);
            }
        });

        // Render series groups first
        const sortedSeriesNames = Object.keys(seriesGroups).sort((a, b) => a.toLowerCase().localeCompare(b.toLowerCase()));

        sortedSeriesNames.forEach(seriesName => {
            const seriesBooks = seriesGroups[seriesName];

            // Create series section
            const seriesSection = document.createElement('div');
            seriesSection.className = 'series-section';

            // Series header
            const header = document.createElement('div');
            header.className = 'series-header';
            header.innerHTML = `<h2>${escapeHtml(seriesName)}</h2><span class="series-count">${seriesBooks.length} book${seriesBooks.length !== 1 ? 's' : ''}</span>`;
            seriesSection.appendChild(header);

            // Books grid for this series
            const grid = document.createElement('div');
            grid.className = 'series-grid';

            seriesBooks.forEach(book => {
                grid.appendChild(createBookCard(book));
            });

            seriesSection.appendChild(grid);
            resultsContainer.appendChild(seriesSection);
        });

        // Render standalone books at the end
        if (standaloneBooks.length > 0) {
            const standaloneSection = document.createElement('div');
            standaloneSection.className = 'series-section';

            const header = document.createElement('div');
            header.className = 'series-header';
            header.innerHTML = `<h2>Standalone Books</h2><span class="series-count">${standaloneBooks.length} book${standaloneBooks.length !== 1 ? 's' : ''}</span>`;
            standaloneSection.appendChild(header);

            const grid = document.createElement('div');
            grid.className = 'series-grid';

            standaloneBooks.forEach(book => {
                grid.appendChild(createBookCard(book));
            });

            standaloneSection.appendChild(grid);
            resultsContainer.appendChild(standaloneSection);
        }

        // Add event listeners to all star ratings
        attachStarListeners();
    }

    function createBookCard(book) {
        const card = document.createElement('div');
        card.className = 'card';
        card.dataset.bookId = book.id;

        const coverUrl = book.id ? `/api/cover/${book.id}` : 'https://via.placeholder.com/300?text=No+Cover';
        const currentRating = book.rating || 0;

        // Show sequence number if part of a series
        const sequenceLabel = book.series && book.series_sequence ? `#${book.series_sequence}` : '';

        card.innerHTML = `
            <img src="${coverUrl}" alt="${escapeHtml(book.title)}" class="card-image" onerror="this.src='https://via.placeholder.com/300?text=No+Cover'">
            <div class="card-content">
                <h3 class="card-title">${sequenceLabel ? `<span class="sequence-badge">${sequenceLabel}</span> ` : ''}${escapeHtml(book.title)}</h3>
                <div class="card-author">by ${escapeHtml(book.author || 'Unknown')}</div>
                <div class="star-rating" data-book-id="${book.id}" data-current-rating="${currentRating}">
                    ${renderStars(currentRating)}
                </div>
            </div>
        `;

        return card;
    }

    function attachStarListeners() {
        document.querySelectorAll('.star-rating').forEach(container => {
            const stars = container.querySelectorAll('.star');

            stars.forEach((star, index) => {
                // Hover effects
                star.addEventListener('mouseenter', () => {
                    highlightStars(container, index + 1);
                });

                star.addEventListener('mouseleave', () => {
                    const currentRating = parseInt(container.dataset.currentRating) || 0;
                    highlightStars(container, currentRating);
                });

                // Click to rate
                star.addEventListener('click', () => {
                    const bookId = container.dataset.bookId;
                    const rating = index + 1;
                    submitRating(bookId, rating, container);
                });
            });
        });
    }

    function renderStars(rating) {
        let html = '';
        for (let i = 1; i <= 5; i++) {
            const filled = i <= rating ? 'filled' : '';
            html += `<span class="star ${filled}" data-star="${i}">★</span>`;
        }
        return html;
    }

    function highlightStars(container, count) {
        const stars = container.querySelectorAll('.star');
        stars.forEach((star, index) => {
            if (index < count) {
                star.classList.add('highlighted');
            } else {
                star.classList.remove('highlighted');
            }
        });
    }

    async function submitRating(bookId, rating, container) {
        try {
            const response = await fetch('/api/rate-book', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ book_id: bookId, rating: rating })
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || 'Failed to save rating');
            }

            // Update the UI
            container.dataset.currentRating = rating;
            const stars = container.querySelectorAll('.star');
            stars.forEach((star, index) => {
                if (index < rating) {
                    star.classList.add('filled');
                } else {
                    star.classList.remove('filled');
                }
            });

            // Update stats
            updateStatsFromDOM();

            // Show success feedback
            container.classList.add('rating-success');
            setTimeout(() => container.classList.remove('rating-success'), 500);

        } catch (error) {
            console.error('Rating error:', error);
            // Show error feedback
            container.classList.add('rating-error');
            setTimeout(() => container.classList.remove('rating-error'), 500);
        }
    }

    function updateStatsFromDOM() {
        const allRatings = document.querySelectorAll('.star-rating');
        let ratedCount = 0;
        allRatings.forEach(container => {
            if (parseInt(container.dataset.currentRating) > 0) {
                ratedCount++;
            }
        });
        ratedCountEl.textContent = `${ratedCount} rated`;
    }

    function escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
});
