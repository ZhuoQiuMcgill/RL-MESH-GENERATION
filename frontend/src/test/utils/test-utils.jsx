import React from 'react'
import { render as rtlRender } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { ApiProvider } from '../../context/ApiProvider'
import { ToastProvider } from '../../shared/ui/Toast'

// Custom render function that includes providers
export function render(
  ui,
  {
    initialEntries = ['/'],
    apiProviderProps = {},
    toastProviderProps = {},
    ...renderOptions
  } = {}
) {
  function Wrapper({ children }) {
    return (
      <MemoryRouter initialEntries={initialEntries}>
        <ApiProvider {...apiProviderProps}>
          <ToastProvider {...toastProviderProps}>
            {children}
          </ToastProvider>
        </ApiProvider>
      </MemoryRouter>
    )
  }

  return rtlRender(ui, { wrapper: Wrapper, ...renderOptions })
}

// Custom render function without providers (for isolated component testing)
export function renderWithoutProviders(ui, options = {}) {
  return rtlRender(ui, options)
}

// Custom render function with just router
export function renderWithRouter(ui, { initialEntries = ['/'], ...renderOptions } = {}) {
  function RouterWrapper({ children }) {
    return (
      <MemoryRouter initialEntries={initialEntries}>
        {children}
      </MemoryRouter>
    )
  }

  return rtlRender(ui, { wrapper: RouterWrapper, ...renderOptions })
}

// Custom render function with just API provider
export function renderWithApi(ui, { apiProviderProps = {}, ...renderOptions } = {}) {
  function ApiWrapper({ children }) {
    return (
      <ApiProvider {...apiProviderProps}>
        {children}
      </ApiProvider>
    )
  }

  return rtlRender(ui, { wrapper: ApiWrapper, ...renderOptions })
}

// Custom render function with just Toast provider
export function renderWithToast(ui, { toastProviderProps = {}, ...renderOptions } = {}) {
  function ToastWrapper({ children }) {
    return (
      <ToastProvider {...toastProviderProps}>
        {children}
      </ToastProvider>
    )
  }

  return rtlRender(ui, { wrapper: ToastWrapper, ...renderOptions })
}

// Helper to create a mock component for testing hooks
export function createHookWrapper(providers = ['api', 'toast', 'router']) {
  return function HookWrapper({ children }) {
    let wrapper = children

    if (providers.includes('toast')) {
      wrapper = <ToastProvider>{wrapper}</ToastProvider>
    }

    if (providers.includes('api')) {
      wrapper = <ApiProvider>{wrapper}</ApiProvider>
    }

    if (providers.includes('router')) {
      wrapper = <MemoryRouter>{wrapper}</MemoryRouter>
    }

    return wrapper
  }
}

// Re-export everything from testing library
export * from '@testing-library/react'
export { default as userEvent } from '@testing-library/user-event'
