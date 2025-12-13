/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{vue,js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                // Audiobookshelf inspired colors
                'brand-dark': '#333333',    // Main Background
                'brand-card': '#444444',    // Card Background
                'brand-header': '#252525',  // Header Background
                'brand-primary': '#4caf50', // ABS Green
                'brand-accent': '#4caf50',  // Mapped to primary for now
            }
        },
    },
    plugins: [],
}
