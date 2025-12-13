import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import path from 'path'

export default defineConfig({
    plugins: [vue()],
    resolve: {
        alias: {
            '@': path.resolve(__dirname, './src'),
        },
    },
    build: {
        outDir: '../web_app/static/dist',
        emptyOutDir: true,
        manifest: true,
        rollupOptions: {
            // Ensure entry point is correct
            input: 'index.html'
        }
    },
    base: '/static/dist/',
    server: {
        proxy: {
            '/api': {
                target: 'http://127.0.0.1:5000',
                changeOrigin: true,
            },
            '/login': {
                target: 'http://127.0.0.1:5000',
                changeOrigin: true,
            },
            '/logout': {
                target: 'http://127.0.0.1:5000',
                changeOrigin: true,
            },
            '/change-password': {
                target: 'http://127.0.0.1:5000',
                changeOrigin: true,
            },
            '/socket.io': {
                target: 'http://127.0.0.1:5000',
                ws: true,
                changeOrigin: true
            }
        }
    }
})
