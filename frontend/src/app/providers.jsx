import React from 'react'
import ApiProvider from '../context/ApiProvider'
import { ThemeProvider, useTheme } from '../core/hooks/useTheme'
import { ToastProvider, ToastContainer } from '../shared/ui'

// Re-export useTheme for backward compatibility
export { useTheme }

// App Providers - Composition of all providers
export function AppProviders({ children }) {
  return (
    <ThemeProvider>
      <ApiProvider>
        <ToastProvider>
          {children}
          <ToastContainer />
        </ToastProvider>
      </ApiProvider>
    </ThemeProvider>
  )
}

export default AppProviders
