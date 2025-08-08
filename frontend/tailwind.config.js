/** @type {import('tailwindcss').Config} */
export default {
  content: ['src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // Reuse primary color variables from old CSS
        primary: {
          50: '#eff6ff',
          100: '#dbeafe', 
          200: '#bfdbfe',
          300: '#93c5fd',
          400: '#60a5fa',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
          800: '#1e40af',
          900: '#1e3a8a',
          // Gradient colors reused from dashboard.css
          start: '#030067',
          end: '#57007c',
        },
        // Reuse gray color variables from old CSS
        gray: {
          50: '#f9fafb',
          100: '#f3f4f6',
          200: '#e5e7eb',
          300: '#d1d5db',
          400: '#9ca3af',
          500: '#6b7280',
          600: '#4b5563',
          700: '#374151',
          800: '#1f2937',
          900: '#111827',
        },
        // Reuse semantic colors
        success: {
          DEFAULT: '#10b981',
          light: 'rgba(16, 185, 129, 0.1)',
          dark: 'rgba(16, 185, 129, 0.2)',
        },
        warning: {
          DEFAULT: '#f59e0b',
          light: 'rgba(245, 158, 11, 0.1)', 
          dark: 'rgba(245, 158, 11, 0.2)',
        },
        error: {
          DEFAULT: '#ef4444',
          light: 'rgba(239, 68, 68, 0.1)',
          dark: 'rgba(239, 68, 68, 0.2)',
        },
        info: {
          DEFAULT: '#60a5fa',
          light: 'rgba(96, 165, 250, 0.1)',
          dark: 'rgba(96, 165, 250, 0.2)',
        },
        // Dark theme colors
        dark: {
          50: '#0f1419',
          100: '#1a1f2e',
          200: '#252a3a', 
          300: '#343a4a',
          400: '#4a5568',
          500: '#718096',
          600: '#a0aec0',
          700: '#cbd5e0',
          800: '#e2e8f0',
          900: '#f7fafc',
        }
      },
      spacing: {
        'xs': '0.25rem',   // 4px
        'sm': '0.5rem',    // 8px 
        'md': '1rem',      // 16px
        'lg': '1.5rem',    // 24px
        'xl': '2rem',      // 32px
        '2xl': '3rem',     // 48px
        '3xl': '4rem',     // 64px
      },
      borderRadius: {
        'sm': '0.125rem',   // 2px
        'md': '0.375rem',   // 6px 
        'lg': '0.5rem',     // 8px
        'xl': '0.75rem',    // 12px
      },
      boxShadow: {
        'sm': '0 1px 2px 0 rgb(0 0 0 / 0.05)',
        'md': '0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1)',
        'lg': '0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1)',
        'xl': '0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1)',
      }
    },
  },
  plugins: [],
}

