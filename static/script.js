document.addEventListener('DOMContentLoaded', async () => {
    const container = document.getElementById('flashcards-container');
    const categoryFilter = document.getElementById('category-filter');
    const flipAllButton = document.getElementById('flip-all');
    const randomFlipButton = document.getElementById('random-flip');
    const resetButton = document.getElementById('reset');

    // Fetch flashcards data
    const response = await fetch('/api/flashcards');
    const flashcards = await response.json();

    // Shuffle function
    const shuffleArray = (array) => {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
    };

    // Shuffle flashcards
    shuffleArray(flashcards);

    // Get unique categories
    const categories = [...new Set(flashcards.map(card => card.category))];
    categories.forEach(category => {
        const option = document.createElement('option');
        option.value = category;
        option.textContent = category;
        categoryFilter.appendChild(option);
    });

    // Render flashcards
    const renderFlashcards = (filteredFlashcards) => {
        container.innerHTML = '';
        filteredFlashcards.forEach(card => {
            const cardElement = document.createElement('div');
            cardElement.className = 'flashcard';
            cardElement.innerHTML = `
                <div class="flashcard-inner">
                    <div class="flashcard-front">
                        <p class="category"><strong>Category:</strong> ${card.category}</p>
                        <p class="definition">${card.definition}</p>
                    </div>
                    <div class="flashcard-back">
                        <h3>${card.term}</h3>
                    </div>
                </div>
            `;

            cardElement.addEventListener('click', () => {
                cardElement.querySelector('.flashcard-inner').classList.toggle('flipped');
            });

            container.appendChild(cardElement);
        });
    };

    // Initial render
    renderFlashcards(flashcards);

    // Filter flashcards based on selected category
    categoryFilter.addEventListener('change', () => {
        const selectedCategory = categoryFilter.value;
        if (selectedCategory === 'all') {
            renderFlashcards(flashcards);
        } else {
            const filteredFlashcards = flashcards.filter(card => card.category === selectedCategory);
            renderFlashcards(filteredFlashcards);
        }
    });

    // Flip all cards
    flipAllButton.addEventListener('click', () => {
        const flashcardInners = document.querySelectorAll('.flashcard-inner');
        flashcardInners.forEach(inner => {
            inner.classList.toggle('flipped');
        });
    });

    // Random flip
    randomFlipButton.addEventListener('click', () => {
        const flashcardInners = document.querySelectorAll('.flashcard-inner');
        flashcardInners.forEach(inner => {
            if (Math.random() > 0.5) {
                inner.classList.add('flipped');
            } else {
                inner.classList.remove('flipped');
            }
        });
    });

    // Reset all cards
    resetButton.addEventListener('click', () => {
        const flashcardInners = document.querySelectorAll('.flashcard-inner');
        flashcardInners.forEach(inner => {
            inner.classList.remove('flipped');
        });
    });
});