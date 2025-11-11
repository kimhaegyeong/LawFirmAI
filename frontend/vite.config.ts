import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    host: '0.0.0.0',
    port: 3000,
    open: true,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
        ws: true,
        configure: (proxy, _options) => {
          proxy.on('proxyReq', (proxyReq, req, _res) => {
            const url = req.url || '';
            if (url.includes('/chat/stream')) {
              proxyReq.setHeader('X-Accel-Buffering', 'no');
              proxyReq.setHeader('Cache-Control', 'no-cache');
              proxyReq.setHeader('Connection', 'keep-alive');
              proxyReq.setHeader('Accept', 'text/event-stream');
            }
          });
          proxy.on('proxyRes', (proxyRes, req, _res) => {
            const url = req.url || '';
            if (url.includes('/chat/stream')) {
              proxyRes.headers['X-Accel-Buffering'] = 'no';
              proxyRes.headers['Cache-Control'] = 'no-cache';
              proxyRes.headers['Connection'] = 'keep-alive';
              proxyRes.headers['Content-Type'] = 'text/event-stream; charset=utf-8';
              delete proxyRes.headers['content-length'];
            }
          });
          proxy.on('error', (err, req, res) => {
            console.error('[Vite Proxy] Error:', err.message);
            if (res && !res.headersSent) {
              res.writeHead(500, {
                'Content-Type': 'text/plain',
              });
              res.end('Proxy error: ' + err.message);
            }
          });
          proxy.on('close', (req, _res) => {
            const url = req.url || '';
            if (url.includes('/chat/stream')) {
              console.log('[Vite Proxy] Stream connection closed:', url);
            }
          });
        },
      },
    },
  },
})

