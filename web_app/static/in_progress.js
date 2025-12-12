document.addEventListener('DOMContentLoaded', () => {
    const resultsContainer = document.getElementById('results');
    const errorMessage = document.getElementById('error-message');
    const loadingContainer = document.getElementById('loading');
    const statsBar = document.getElementById('stats');
    const bookCountEl = document.getElementById('book-count');

    // Load in-progress books on page load
    loadInProgressBooks();

    async function loadInProgressBooks() {
        try {
            const response = await fetch('/api/in-progress');

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || 'Failed to fetch in-progress books');
            }

            const books = await response.json();

            if (books.length === 0) {
                loadingContainer.classList.add('hidden');
                errorMessage.textContent = 'No in-progress audiobooks found. Start listening!';
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
        bookCountEl.textContent = `${books.length} book${books.length !== 1 ? 's' : ''}`;
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
    }

    function createBookCard(book) {
        const card = document.createElement('div');
        card.className = 'card';
        card.style.cursor = 'default'; // No click action on card for now

        // Use proxy cover endpoint by default to avoid CORS/auth issues with direct ABS links if on client
        const coverUrl = book.id ? `/api/cover/${book.id}` : 'https://via.placeholder.com/300?text=No+Cover';

        // Show sequence number if part of a series
        const sequenceLabel = book.series && book.series_sequence ? `#${book.series_sequence}` : '';

        // Progress bar logic
        const progressPercent = book.progress ? Math.round(book.progress * 100) : 0;
        const progressBar = `
            <div class="progress-container" style="background-color: #334155; height: 6px; width: 100%; overflow: hidden;">
                <div class="progress-bar" style="background-color: #4caf50; height: 100%; width: ${progressPercent}%;"></div>
            </div>
            <!-- Optional: Percentage text overlay or separate line? 
                 User asked for 'right under', usually implies just the bar. 
                 I'll remove the text for a cleaner look or keep it small. 
                 User said "same width", text might break flow if outside.
                 I'll keep just the bar for now as per "make it the same width" focus. -->
        `;

        card.innerHTML = `
            <img src="${coverUrl}" alt="${escapeHtml(book.title)}" class="card-image" onerror="this.src='https://via.placeholder.com/300?text=No+Cover'">
            ${progressBar}
            <div class="card-content">
                <h3 class="card-title">${sequenceLabel ? `<span class="sequence-badge">${sequenceLabel}</span> ` : ''}${escapeHtml(book.title)}</h3>
                <div class="card-author">by ${escapeHtml(book.author || 'Unknown')}</div>
            </div>
        `;

        return card;
    }

    function escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
});
