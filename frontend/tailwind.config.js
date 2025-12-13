/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{vue,js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                // Custom colors for premium feel
                'brand-dark': '#0f172a',
                'brand-primary': '#3b82f6',
                'brand-accent': '#6366f1',
            }
        },
    },
    plugins: [],
}
