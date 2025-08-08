import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { Suspense } from 'react'
import { routes } from './app/routes'
import AppProviders from './app/providers'
import AppShell from './shared/layout/AppShell'
import { ErrorBoundary } from './shared/ui'
import './App.css'

// Loading fallback component
const LoadingFallback = () => (
  <div className="flex items-center justify-center min-h-[400px]">
    <div className="text-center">
      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-4"></div>
      <p className="text-text-secondary">Loading...</p>
    </div>
  </div>
)

// App component wrapped with theme context
function AppContent() {
  return (
    <Router>
      <AppShell 
        showSidebar={false}
      >
        <ErrorBoundary 
          showReload={true}
          errorMessage="Sorry, this page encountered an unexpected error."
          onError={(error, errorInfo) => {
            console.error('Route Error:', error, errorInfo)
            // Could send to error reporting service here
          }}
        >
          <Suspense fallback={<LoadingFallback />}>
            <Routes>
              {routes.map((route) => {
                const Component = route.element
                return (
                  <Route
                    key={route.path}
                    path={route.path}
                    element={
                      <ErrorBoundary
                        errorMessage={`Error loading ${route.title}`}
                        className="min-h-[300px]"
                      >
                        <Component />
                      </ErrorBoundary>
                    }
                  />
                )
              })}
            </Routes>
          </Suspense>
        </ErrorBoundary>
      </AppShell>
    </Router>
  )
}

function App() {
  return (
    <AppProviders>
      <AppContent />
    </AppProviders>
  )
}

export default App
