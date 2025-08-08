import React from 'react'
import { Button } from './index'
import { cn } from '../utils/cn'

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props)
    this.state = { 
      hasError: false, 
      error: null, 
      errorInfo: null,
      retryCount: 0
    }
  }

  static getDerivedStateFromError(error) {
    // Update state so the next render will show the fallback UI
    return { hasError: true }
  }

  componentDidCatch(error, errorInfo) {
    // Log error details
    console.error('ErrorBoundary caught an error:', error, errorInfo)
    
    // Update state with error details
    this.setState({
      error,
      errorInfo,
      hasError: true
    })

    // Report to error reporting service if available
    if (this.props.onError) {
      this.props.onError(error, errorInfo)
    }
  }

  handleRetry = () => {
    this.setState(prevState => ({
      hasError: false,
      error: null,
      errorInfo: null,
      retryCount: prevState.retryCount + 1
    }))
  }

  render() {
    if (this.state.hasError) {
      // Custom fallback UI
      if (this.props.fallback) {
        return this.props.fallback(this.state.error, this.handleRetry)
      }

      // Default error UI
      const isDev = import.meta.env.DEV

      return (
        <div className={cn(
          "flex flex-col items-center justify-center min-h-[400px] p-8 text-center",
          "bg-bg-primary border border-border-custom rounded-lg",
          this.props.className
        )}>
          <div className="mb-6">
            <div className="w-16 h-16 mx-auto mb-4 flex items-center justify-center bg-red-100 rounded-full">
              <svg 
                className="w-8 h-8 text-red-600" 
                fill="none" 
                stroke="currentColor" 
                viewBox="0 0 24 24"
              >
                <path 
                  strokeLinecap="round" 
                  strokeLinejoin="round" 
                  strokeWidth={2} 
                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" 
                />
              </svg>
            </div>
            <h2 className="text-xl font-semibold text-text-primary mb-2">
              Something went wrong
            </h2>
            <p className="text-text-secondary max-w-md">
              {this.props.errorMessage || "An unexpected error occurred. Please try again."}
            </p>
          </div>

          <div className="flex flex-col sm:flex-row gap-3 mb-6">
            <Button 
              onClick={this.handleRetry}
              variant="primary"
              className="min-w-[120px]"
            >
              Try Again
            </Button>
            
            {this.props.showReload && (
              <Button 
                onClick={() => window.location.reload()}
                variant="outline"
                className="min-w-[120px]"
              >
                Reload Page
              </Button>
            )}
            
            {this.props.onReset && (
              <Button 
                onClick={this.props.onReset}
                variant="secondary"
                className="min-w-[120px]"
              >
                Reset
              </Button>
            )}
          </div>

          {isDev && this.state.error && (
            <details className="mt-4 max-w-2xl w-full">
              <summary className="cursor-pointer text-sm text-text-secondary hover:text-text-primary mb-2">
                Show Error Details (Development)
              </summary>
              <div className="bg-gray-100 border rounded-lg p-4 text-left text-sm font-mono overflow-auto max-h-64">
                <div className="text-red-600 font-semibold mb-2">Error:</div>
                <div className="mb-4">{this.state.error.toString()}</div>
                
                {this.state.errorInfo && (
                  <>
                    <div className="text-red-600 font-semibold mb-2">Component Stack:</div>
                    <pre className="whitespace-pre-wrap text-xs">
                      {this.state.errorInfo.componentStack}
                    </pre>
                  </>
                )}
              </div>
            </details>
          )}

          {this.state.retryCount > 0 && (
            <div className="mt-2 text-xs text-text-secondary">
              Retry attempts: {this.state.retryCount}
            </div>
          )}
        </div>
      )
    }

    return this.props.children
  }
}

// HOC for wrapping components with error boundary
export const withErrorBoundary = (Component, errorBoundaryProps = {}) => {
  const WrappedComponent = React.forwardRef((props, ref) => (
    <ErrorBoundary {...errorBoundaryProps}>
      <Component {...props} ref={ref} />
    </ErrorBoundary>
  ))
  
  WrappedComponent.displayName = `withErrorBoundary(${Component.displayName || Component.name})`
  
  return WrappedComponent
}

ErrorBoundary.displayName = 'ErrorBoundary'

export default ErrorBoundary
