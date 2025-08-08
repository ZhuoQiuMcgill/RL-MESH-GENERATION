import { describe, it, expect, vi, beforeEach } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { renderWithoutProviders as render } from '../../../../test/utils/test-utils'
import Toast from '../Toast'
import { TOAST_TYPES } from '../ToastContext'

// Mock toast object factory
const createMockToast = (overrides = {}) => ({
  id: 'test-toast-1',
  type: TOAST_TYPES.INFO,
  message: 'Test message',
  title: null,
  description: null,
  icon: null,
  duration: 4000,
  pauseOnHover: true,
  showCloseButton: true,
  action: null,
  ...overrides
})

describe('Toast', () => {
  let mockOnRemove

  beforeEach(() => {
    mockOnRemove = vi.fn()
  })

  it('renders basic toast message', () => {
    const toast = createMockToast()
    
    render(<Toast toast={toast} onRemove={mockOnRemove} />)
    
    expect(screen.getByText('Test message')).toBeInTheDocument()
    expect(screen.getByRole('alert')).toBeInTheDocument()
  })

  it('renders toast with title', () => {
    const toast = createMockToast({
      title: 'Test Title',
      message: 'Test message content'
    })
    
    render(<Toast toast={toast} onRemove={mockOnRemove} />)
    
    expect(screen.getByText('Test Title')).toBeInTheDocument()
    expect(screen.getByText('Test message content')).toBeInTheDocument()
  })

  it('renders toast with description', () => {
    const toast = createMockToast({
      message: 'Main message',
      description: 'Additional details'
    })
    
    render(<Toast toast={toast} onRemove={mockOnRemove} />)
    
    expect(screen.getByText('Main message')).toBeInTheDocument()
    expect(screen.getByText('Additional details')).toBeInTheDocument()
  })

  it('applies correct styles for different toast types', () => {
    const { rerender } = render(
      <Toast toast={createMockToast({ type: TOAST_TYPES.INFO })} onRemove={mockOnRemove} />
    )
    
    let toastElement = screen.getByRole('alert')
    expect(toastElement).toHaveClass('bg-blue-50', 'border-blue-300', 'text-blue-800')
    
    rerender(
      <Toast toast={createMockToast({ type: TOAST_TYPES.SUCCESS })} onRemove={mockOnRemove} />
    )
    toastElement = screen.getByRole('alert')
    expect(toastElement).toHaveClass('bg-green-50', 'border-green-300', 'text-green-800')
    
    rerender(
      <Toast toast={createMockToast({ type: TOAST_TYPES.WARNING })} onRemove={mockOnRemove} />
    )
    toastElement = screen.getByRole('alert')
    expect(toastElement).toHaveClass('bg-yellow-50', 'border-yellow-300', 'text-yellow-800')
    
    rerender(
      <Toast toast={createMockToast({ type: TOAST_TYPES.ERROR })} onRemove={mockOnRemove} />
    )
    toastElement = screen.getByRole('alert')
    expect(toastElement).toHaveClass('bg-red-50', 'border-red-300', 'text-red-800')
  })

  it('shows correct icons for different toast types', () => {
    const { rerender } = render(
      <Toast toast={createMockToast({ type: TOAST_TYPES.INFO })} onRemove={mockOnRemove} />
    )
    
    expect(screen.getByRole('alert').querySelector('svg')).toBeInTheDocument()
    
    rerender(
      <Toast toast={createMockToast({ type: TOAST_TYPES.SUCCESS })} onRemove={mockOnRemove} />
    )
    expect(screen.getByRole('alert').querySelector('svg')).toBeInTheDocument()
    
    rerender(
      <Toast toast={createMockToast({ type: TOAST_TYPES.WARNING })} onRemove={mockOnRemove} />
    )
    expect(screen.getByRole('alert').querySelector('svg')).toBeInTheDocument()
    
    rerender(
      <Toast toast={createMockToast({ type: TOAST_TYPES.ERROR })} onRemove={mockOnRemove} />
    )
    expect(screen.getByRole('alert').querySelector('svg')).toBeInTheDocument()
  })

  it('hides icon when icon prop is false', () => {
    const toast = createMockToast({ icon: false })
    
    render(<Toast toast={toast} onRemove={mockOnRemove} />)
    
    expect(screen.getByRole('alert').querySelector('svg')).not.toBeInTheDocument()
  })

  it('shows custom icon when provided', () => {
    const customIcon = <span data-testid="custom-icon">🎉</span>
    const toast = createMockToast({ icon: customIcon })
    
    render(<Toast toast={toast} onRemove={mockOnRemove} />)
    
    expect(screen.getByTestId('custom-icon')).toBeInTheDocument()
  })

  it('shows close button when showCloseButton is true', () => {
    const toast = createMockToast({ showCloseButton: true })
    
    render(<Toast toast={toast} onRemove={mockOnRemove} />)
    
    const closeButton = screen.getByLabelText('Close notification')
    expect(closeButton).toBeInTheDocument()
  })

  it('hides close button when showCloseButton is false', () => {
    const toast = createMockToast({ showCloseButton: false })
    
    render(<Toast toast={toast} onRemove={mockOnRemove} />)
    
    const closeButton = screen.queryByLabelText('Close notification')
    expect(closeButton).not.toBeInTheDocument()
  })

  it('calls onRemove when close button is clicked', async () => {
    const user = userEvent.setup()
    const toast = createMockToast({ showCloseButton: true })
    
    render(<Toast toast={toast} onRemove={mockOnRemove} />)
    
    const closeButton = screen.getByLabelText('Close notification')
    await user.click(closeButton)
    
    // Should call onRemove after animation delay
    await waitFor(() => {
      expect(mockOnRemove).toHaveBeenCalledWith(toast.id)
    }, { timeout: 500 })
  })

  it('shows action button when action is provided', () => {
    const toast = createMockToast({
      action: {
        label: 'Undo',
        onClick: vi.fn()
      }
    })
    
    render(<Toast toast={toast} onRemove={mockOnRemove} />)
    
    const actionButton = screen.getByRole('button', { name: 'Undo' })
    expect(actionButton).toBeInTheDocument()
  })

  it('calls action onClick when action button is clicked', async () => {
    const user = userEvent.setup()
    const actionOnClick = vi.fn()
    const toast = createMockToast({
      action: {
        label: 'Retry',
        onClick: actionOnClick
      }
    })
    
    render(<Toast toast={toast} onRemove={mockOnRemove} />)
    
    const actionButton = screen.getByRole('button', { name: 'Retry' })
    await user.click(actionButton)
    
    expect(actionOnClick).toHaveBeenCalledTimes(1)
  })

  it('closes toast after action click by default', async () => {
    const user = userEvent.setup()
    const toast = createMockToast({
      action: {
        label: 'Action',
        onClick: vi.fn()
      }
    })
    
    render(<Toast toast={toast} onRemove={mockOnRemove} />)
    
    const actionButton = screen.getByRole('button', { name: 'Action' })
    await user.click(actionButton)
    
    // Should call onRemove after animation delay
    await waitFor(() => {
      expect(mockOnRemove).toHaveBeenCalledWith(toast.id)
    }, { timeout: 500 })
  })

  it('does not close toast after action click when closeOnClick is false', async () => {
    const user = userEvent.setup()
    const toast = createMockToast({
      action: {
        label: 'Action',
        onClick: vi.fn(),
        closeOnClick: false
      }
    })
    
    render(<Toast toast={toast} onRemove={mockOnRemove} />)
    
    const actionButton = screen.getByRole('button', { name: 'Action' })
    await user.click(actionButton)
    
    // Wait a bit to ensure onRemove is not called
    await new Promise(resolve => setTimeout(resolve, 400))
    expect(mockOnRemove).not.toHaveBeenCalled()
  })

  it('auto-dismisses after duration', async () => {
    const toast = createMockToast({ duration: 100 }) // Short duration for testing
    
    render(<Toast toast={toast} onRemove={mockOnRemove} />)
    
    // Should auto-dismiss after duration
    await waitFor(() => {
      expect(mockOnRemove).toHaveBeenCalledWith(toast.id)
    }, { timeout: 500 })
  })

  it('does not auto-dismiss when duration is 0', async () => {
    const toast = createMockToast({ duration: 0 })
    
    render(<Toast toast={toast} onRemove={mockOnRemove} />)
    
    // Wait longer than normal auto-dismiss time
    await new Promise(resolve => setTimeout(resolve, 200))
    expect(mockOnRemove).not.toHaveBeenCalled()
  })

  it('pauses auto-dismiss on hover when pauseOnHover is true', async () => {
    const user = userEvent.setup()
    const toast = createMockToast({ 
      duration: 200, // Short duration for testing
      pauseOnHover: true 
    })
    
    render(<Toast toast={toast} onRemove={mockOnRemove} />)
    
    const toastElement = screen.getByRole('alert')
    
    // Hover over toast to pause
    await user.hover(toastElement)
    
    // Wait longer than duration but should not dismiss while hovered
    await new Promise(resolve => setTimeout(resolve, 250))
    expect(mockOnRemove).not.toHaveBeenCalled()
    
    // Unhover to resume
    await user.unhover(toastElement)
    
    // Now should dismiss
    await waitFor(() => {
      expect(mockOnRemove).toHaveBeenCalledWith(toast.id)
    }, { timeout: 300 })
  })

  it('shows progress bar for timed toasts', () => {
    const toast = createMockToast({ duration: 4000 })
    
    render(<Toast toast={toast} onRemove={mockOnRemove} />)
    
    const progressBar = screen.getByRole('alert').querySelector('.absolute.bottom-0')
    expect(progressBar).toBeInTheDocument()
    expect(progressBar).toHaveClass('h-1', 'bg-current', 'opacity-30')
  })

  it('hides progress bar for permanent toasts', () => {
    const toast = createMockToast({ duration: 0 })
    
    render(<Toast toast={toast} onRemove={mockOnRemove} />)
    
    const progressBar = screen.getByRole('alert').querySelector('.absolute.bottom-0')
    expect(progressBar).not.toBeInTheDocument()
  })

  it('applies custom className', () => {
    const toast = createMockToast()
    
    render(<Toast toast={toast} onRemove={mockOnRemove} className="custom-toast" />)
    
    const toastElement = screen.getByRole('alert')
    expect(toastElement).toHaveClass('custom-toast')
  })

  it('handles JSX message content', () => {
    const toast = createMockToast({
      message: <span data-testid="jsx-message">JSX Message</span>
    })
    
    render(<Toast toast={toast} onRemove={mockOnRemove} />)
    
    expect(screen.getByTestId('jsx-message')).toBeInTheDocument()
  })

  it('has correct ARIA attributes', () => {
    const toast = createMockToast()
    
    render(<Toast toast={toast} onRemove={mockOnRemove} />)
    
    const toastElement = screen.getByRole('alert')
    expect(toastElement).toHaveAttribute('role', 'alert')
    expect(toastElement).toHaveAttribute('aria-live', 'polite')
  })
})
