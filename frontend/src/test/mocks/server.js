import { setupServer } from 'msw/node'
import { handlers } from './handlers.js'

// Setup the mock server with default handlers
export const server = setupServer(...handlers)

// Start server before all tests
export const startMockServer = () => {
  server.listen({ onUnhandledRequest: 'error' })
}

// Reset handlers after each test
export const resetMockServer = () => {
  server.resetHandlers()
}

// Close server after all tests
export const closeMockServer = () => {
  server.close()
}

// Helper function to override handlers for specific tests
export const overrideHandlers = (...newHandlers) => {
  server.use(...newHandlers)
}

export default server
