document.addEventListener('DOMContentLoaded', () => {
    const generateBtn = document.getElementById('generate-btn');
    const resultsContainer = document.getElementById('results');
    const errorMessage = document.getElementById('error-message');
    const loader = generateBtn.querySelector('.loader');
    const btnText = generateBtn.querySelector('.btn-text');

    async function fetchRecommendations(refresh = false) {
        // Reset state if refreshing
        if (refresh) {
            resultsContainer.innerHTML = '';
            resultsContainer.classList.add('hidden');
            errorMessage.classList.add('hidden');
        }

        // Loading state
        generateBtn.disabled = true;
        if (refresh) {
            loader.classList.remove('hidden');
            btnText.textContent = 'Generating...';
        }

        try {
            const url = refresh ? '/api/recommend?refresh=true' : '/api/recommend';
            const response = await fetch(url);

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || 'Failed to fetch recommendations');
            }

            const data = await response.json();
            
            // Handle response structure { recommendations: [], generated_at: "" } or fallback
            let recommendations = [];
            let generatedAt = null;

            if (Array.isArray(data)) {
                recommendations = data;
            } else if (data.recommendations) {
                recommendations = data.recommendations;
                generatedAt = data.generated_at;
            }

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
            errorMessage.textContent = error.message;
            errorMessage.classList.remove('hidden');
        } finally {
            // Reset button
            generateBtn.disabled = false;
            loader.classList.add('hidden');
            btnText.textContent = 'Generate Recommendations';
        }
    }

    generateBtn.addEventListener('click', () => fetchRecommendations(true));
    
    // Initial fetch on load
    fetchRecommendations(false);

    function renderRecommendations(books) {
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
