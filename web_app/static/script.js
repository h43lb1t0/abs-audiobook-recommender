document.addEventListener('DOMContentLoaded', () => {
    const generateBtn = document.getElementById('generate-btn');
    const resultsContainer = document.getElementById('results');
    const errorMessage = document.getElementById('error-message');
    const loader = generateBtn.querySelector('.loader');
    const btnText = generateBtn.querySelector('.btn-text');

    // Initialize SocketIO
    const socket = io();

    // Check for last recommendations
    fetchLastRecommendations();

    async function fetchLastRecommendations() {
        try {
            const response = await fetch('/api/last_recommendations');
            if (response.ok) {
                const recommendations = await response.json();
                if (recommendations.length > 0) {
                    renderRecommendations(recommendations);
                    resultsContainer.classList.remove('hidden');
                }
            }
        } catch (error) {
            console.error('Failed to fetch last recommendations:', error);
        }
    }

    // Socket Event Listeners

    socket.on('connect', () => {
        console.log('Connected to server via WebSocket');
    });

    socket.on('status', (data) => {
        console.log('Status:', data.message);
        btnText.textContent = data.message;
    });

    socket.on('recommendations', (recommendations) => {
        // Reset UI state
        generateBtn.disabled = false;
        loader.classList.add('hidden');
        btnText.textContent = 'Generate Recommendations';

        if (recommendations.length === 0) {
            errorMessage.textContent = 'No recommendations found. Check your library or API key.';
            errorMessage.classList.remove('hidden');
            return;
        }

        renderRecommendations(recommendations);
        resultsContainer.classList.remove('hidden');
    });

    socket.on('error', (data) => {
        console.error('Socket error:', data.message);
        errorMessage.textContent = data.message;
        errorMessage.classList.remove('hidden');

        // Reset button
        generateBtn.disabled = false;
        loader.classList.add('hidden');
        btnText.textContent = 'Generate Recommendations';
    });

    generateBtn.addEventListener('click', async () => {
        // Reset state
        resultsContainer.innerHTML = '';
        resultsContainer.classList.add('hidden');
        errorMessage.classList.add('hidden');

        // Loading state
        generateBtn.disabled = true;
        loader.classList.remove('hidden');
        btnText.textContent = 'Initializing...';

        // Emit event to start generation
        socket.emit('generate_recommendations');
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
