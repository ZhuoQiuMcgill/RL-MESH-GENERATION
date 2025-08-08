/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      // Colors using CSS custom properties for theme consistency
      colors: {
        'primary-start': 'var(--color-primary-start)',
        'primary-end': 'var(--color-primary-end)',
        'bg-primary': 'var(--color-bg-primary)',
        'bg-secondary': 'var(--color-bg-secondary)',
        'text-primary': 'var(--color-text-primary)',
        'text-secondary': 'var(--color-text-secondary)',
        'accent': 'var(--color-accent)',
        'border-custom': 'var(--color-border)',
        'card': 'var(--color-bg-card)',
        'card-hover': 'var(--color-bg-card-hover)',
        // Semantic colors
        'success': 'var(--color-success)',
        'warning': 'var(--color-warning)',
        'error': 'var(--color-error)',
        'info': 'var(--color-info)',
      },
      
      // Spacing scale using design tokens
      spacing: {
        'xs': 'var(--space-xs)',
        'sm': 'var(--space-sm)',
        'md': 'var(--space-md)',
        'lg': 'var(--space-lg)',
        'xl': 'var(--space-xl)',
        '2xl': 'var(--space-2xl)',
        '3xl': 'var(--space-3xl)',
        '4xl': 'var(--space-4xl)',
      },
      
      // Border radius using design tokens
      borderRadius: {
        'sm': 'var(--radius-sm)',
        'md': 'var(--radius-md)',
        'lg': 'var(--radius-lg)',
        'xl': 'var(--radius-xl)',
      },
      
      // Container max-widths
      maxWidth: {
        'sm': 'var(--container-sm)',
        'md': 'var(--container-md)',
        'lg': 'var(--container-lg)',
        'xl': 'var(--container-xl)',
        '2xl': 'var(--container-2xl)',
      },
      
      // Font family
      fontFamily: {
        'sans': ['var(--font-family-sans)'],
        'mono': ['var(--font-family-mono)'],
      },
      
      // Font sizes using design tokens
      fontSize: {
        'xs': 'var(--text-xs)',
        'sm': 'var(--text-sm)',
        'base': 'var(--text-base)',
        'lg': 'var(--text-lg)',
        'xl': 'var(--text-xl)',
        '2xl': 'var(--text-2xl)',
        '3xl': 'var(--text-3xl)',
        '4xl': 'var(--text-4xl)',
      },
      
      // Box shadows
      boxShadow: {
        'sm': 'var(--shadow-sm)',
        'md': 'var(--shadow-md)',
        'lg': 'var(--shadow-lg)',
        'xl': 'var(--shadow-xl)',
      },
      
      // Background images with design tokens
      backgroundImage: {
        'primary-gradient': 'linear-gradient(135deg, var(--color-primary-start), var(--color-primary-end))',
      },
      
      // Transitions
      transitionDuration: {
        'fast': 'var(--transition-fast)',
        'normal': 'var(--transition-normal)',
        'slow': 'var(--transition-slow)',
      },
    },
  },
  plugins: [],
}
