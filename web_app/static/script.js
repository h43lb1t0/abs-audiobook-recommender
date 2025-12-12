document.addEventListener('DOMContentLoaded', () => {
    const socket = io();
    const generateBtn = document.getElementById('generate-btn');
    const resultsContainer = document.getElementById('results');
    const errorMessage = document.getElementById('error-message');
    const loader = generateBtn.querySelector('.loader');
    const btnText = generateBtn.querySelector('.btn-text');

    function setLoading(isLoading) {
        if (isLoading) {
            generateBtn.disabled = true;
            loader.classList.remove('hidden');
            btnText.textContent = 'Generating...';
            errorMessage.classList.add('hidden');
        } else {
            generateBtn.disabled = false;
            loader.classList.add('hidden');
            btnText.textContent = 'Generate Recommendations';
        }
    }

    // Listen for recommendations
    socket.on('recommendations_ready', (data) => {
        setLoading(false);
        try {
            const recommendations = data.recommendations || [];
            const generatedAt = data.generated_at;

            if (recommendations.length === 0) {
                throw new Error('No recommendations found. Check your library or API key.');
            }

            renderRecommendations(recommendations);

            const lastUpdatedEl = document.getElementById('last-updated');
            if (generatedAt) {
                const date = new Date(generatedAt);
                lastUpdatedEl.textContent = `Last generated: ${date.toLocaleString(undefined, {
                    year: 'numeric',
                    month: 'numeric',
                    day: 'numeric',
                    hour: 'numeric',
                    minute: '2-digit'
                })}`;
                lastUpdatedEl.classList.remove('hidden');
            } else {
                lastUpdatedEl.classList.add('hidden');
            }

            resultsContainer.classList.remove('hidden');
        } catch (error) {
            showError(error.message);
        }
    });

    // Listen for errors
    socket.on('error', (data) => {
        setLoading(false);
        showError(data.error || 'An error occurred');
    });

    function showError(msg) {
        errorMessage.textContent = msg;
        errorMessage.classList.remove('hidden');
        resultsContainer.classList.add('hidden');
    }

    function fetchRecommendations(refresh = false) {
        // Reset state if refreshing
        if (refresh) {
            resultsContainer.innerHTML = '';
            resultsContainer.classList.add('hidden');
            errorMessage.classList.add('hidden');
        }

        setLoading(true);
        socket.emit('get_recommendations', { refresh: refresh });
    }

    const forceSyncBtn = document.getElementById('force-sync-btn');
    if (forceSyncBtn) {
        forceSyncBtn.addEventListener('click', async () => {
            if (!confirm('Are you sure you want to force sync the library? This might take a while.')) return;

            const originalText = forceSyncBtn.textContent;
            forceSyncBtn.disabled = true;
            forceSyncBtn.textContent = 'Syncing...';

            try {
                const response = await fetch('/api/admin/force-sync', { method: 'POST' });
                const data = await response.json();

                if (response.ok) {
                    alert('Sync triggered successfully!');
                } else {
                    alert('Error: ' + (data.error || 'Unknown error'));
                }
            } catch (err) {
                alert('Request failed: ' + err.message);
            } finally {
                forceSyncBtn.disabled = false;
                forceSyncBtn.textContent = originalText;
            }
        });
    }

    generateBtn.addEventListener('click', () => fetchRecommendations(true));

    // Initial fetch on load
    fetchRecommendations(false);

    function renderRecommendations(books) {
        resultsContainer.innerHTML = ''; // Clear previous results
        books.forEach(book => {
            const card = document.createElement('div');
            card.className = 'card';

            const coverUrl = book.id ? `/api/cover/${book.id}` : 'https://via.placeholder.com/300?text=No+Cover';

            card.innerHTML = `
                <img src="${coverUrl}" alt="${book.title}" class="card-image" onerror="this.src='https://via.placeholder.com/300?text=No+Cover'">
                <div class="card-content">
                    <h3 class="card-title">${book.title}</h3>
                    <div class="card-author">by ${book.author}</div>
                    <div class="card-reason">${book.reason}</div>
                </div>
            `;

            resultsContainer.appendChild(card);
        });
    }
});
