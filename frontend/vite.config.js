import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { visualizer } from 'rollup-plugin-visualizer'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    // Bundle analyzer - only in build mode
    visualizer({
      filename: 'dist/bundle-analysis.html',
      open: false,
      gzipSize: true,
      brotliSize: true,
    })
  ],
  
  // Performance optimizations
  build: {
    // Optimize chunk splitting for better caching
    rollupOptions: {
      output: {
        manualChunks: {
          // Vendor chunk for React and core libraries
          vendor: ['react', 'react-dom'],
          // Router chunk for React Router
          router: ['react-router-dom'],
          // Icons chunk for Lucide icons
          icons: ['lucide-react'],
          // Canvas utilities chunk
          canvas: ['./src/utils/CanvasRenderer.js', './src/components/MeshCanvas.jsx']
        }
      }
    },
    // Enable source maps for better debugging in production
    sourcemap: false,
    // Optimize CSS
    cssCodeSplit: true,
    // Set reasonable chunk size warning limit
    chunkSizeWarningLimit: 600,
    // Enable minification
    minify: 'esbuild',
    // Target modern browsers for better optimization
    target: 'es2020'
  },
  
  // Development server optimizations
  server: {
    // Enable HTTP/2 for faster development
    https: false,
    // Optimize HMR
    hmr: {
      overlay: true
    }
  },
  
  // Preview server optimizations (for production builds)
  preview: {
    port: 4173,
    host: true,
    // Add cache headers for better performance
    headers: {
      // Cache static assets for 1 year
      'Cache-Control': 'public, max-age=31536000, immutable',
    },
    // Enable compression
    cors: true,
  },
  
  // Optimize dependency pre-bundling
  optimizeDeps: {
    include: [
      'react',
      'react-dom',
      'react-router-dom',
      'lucide-react'
    ],
    // Force re-optimization on these dependencies
    force: false
  },
  
  // Enable CSS preprocessing optimizations
  css: {
    // Enable CSS modules for better encapsulation
    modules: {
      localsConvention: 'camelCase'
    },
    // Optimize CSS processing
    devSourcemap: true
  }
})
