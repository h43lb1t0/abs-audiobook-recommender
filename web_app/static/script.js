document.addEventListener('DOMContentLoaded', () => {
    const generateBtn = document.getElementById('generate-btn');
    const resultsContainer = document.getElementById('results');
    const errorMessage = document.getElementById('error-message');
    const loader = generateBtn.querySelector('.loader');
    const btnText = generateBtn.querySelector('.btn-text');

    generateBtn.addEventListener('click', async () => {
        // Reset state
        resultsContainer.innerHTML = '';
        resultsContainer.classList.add('hidden');
        errorMessage.classList.add('hidden');

        // Loading state
        generateBtn.disabled = true;
        loader.classList.remove('hidden');
        btnText.textContent = 'Generating...';

        try {
            const response = await fetch('/api/recommend');

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || 'Failed to fetch recommendations');
            }

            const recommendations = await response.json();

            if (recommendations.length === 0) {
                throw new Error('No recommendations found. Check your library or API key.');
            }

            renderRecommendations(recommendations);
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
    });

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
