import { createContext, useContext, useState, useEffect } from 'react'

// Theme Context
const ThemeContext = createContext(null)

// Theme preferences
const THEME_OPTIONS = {
  LIGHT: 'light',
  DARK: 'dark',
  SYSTEM: 'system'
}

// Theme Provider Component with enhanced functionality
export function ThemeProvider({ children }) {
  // Get system preference
  const getSystemPreference = () => {
    if (typeof window === 'undefined') return false
    return window.matchMedia('(prefers-color-scheme: dark)').matches
  }

  // Initialize theme preference from localStorage or default to system
  const [themePreference, setThemePreference] = useState(() => {
    if (typeof window === 'undefined') return THEME_OPTIONS.SYSTEM
    
    // Check for new preference key first
    const saved = localStorage.getItem('themePreference')
    if (saved) return saved
    
    // Migrate old isDarkMode setting
    const oldSetting = localStorage.getItem('isDarkMode')
    if (oldSetting !== null) {
      const wasDark = JSON.parse(oldSetting)
      const migratedPreference = wasDark ? THEME_OPTIONS.DARK : THEME_OPTIONS.LIGHT
      // Clean up old key
      localStorage.removeItem('isDarkMode')
      // Save migrated preference
      localStorage.setItem('themePreference', migratedPreference)
      return migratedPreference
    }
    
    return THEME_OPTIONS.SYSTEM
  })

  // Calculate actual dark mode state based on preference
  const [systemDark, setSystemDark] = useState(getSystemPreference)

  const isDark = themePreference === THEME_OPTIONS.SYSTEM 
    ? systemDark
    : themePreference === THEME_OPTIONS.DARK

  // Listen for system theme changes
  useEffect(() => {
    if (typeof window === 'undefined') return

    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
    const handleChange = (e) => {
      setSystemDark(e.matches)
    }

    mediaQuery.addEventListener('change', handleChange)
    return () => mediaQuery.removeEventListener('change', handleChange)
  }, [])

  // Apply theme to document root and save to localStorage
  useEffect(() => {
    const root = document.documentElement
    
    if (isDark) {
      root.classList.add('dark')
      root.classList.remove('light')
    } else {
      root.classList.add('light')
      root.classList.remove('dark')
    }
    
    // Save preference to localStorage
    localStorage.setItem('themePreference', themePreference)
  }, [isDark, themePreference])

  // Theme actions
  const setTheme = (preference) => {
    if (Object.values(THEME_OPTIONS).includes(preference)) {
      setThemePreference(preference)
    }
  }

  const toggleTheme = () => {
    if (themePreference === THEME_OPTIONS.LIGHT) {
      setThemePreference(THEME_OPTIONS.DARK)
    } else if (themePreference === THEME_OPTIONS.DARK) {
      setThemePreference(THEME_OPTIONS.SYSTEM)
    } else {
      setThemePreference(THEME_OPTIONS.LIGHT)
    }
  }

  const contextValue = {
    // Current state
    isDark,
    themePreference,
    systemDark,
    
    // Actions
    setTheme,
    toggleTheme,
    
    // Utilities
    theme: isDark ? 'dark' : 'light',
    isSystem: themePreference === THEME_OPTIONS.SYSTEM,
    
    // Constants for consumers
    THEME_OPTIONS
  }

  return (
    <ThemeContext.Provider value={contextValue}>
      {children}
    </ThemeContext.Provider>
  )
}

// Enhanced useTheme Hook
export function useTheme() {
  const context = useContext(ThemeContext)
  
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider')
  }
  
  return context
}

// Export theme constants for use elsewhere
export { THEME_OPTIONS }

export default useTheme
