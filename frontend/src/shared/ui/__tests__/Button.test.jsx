import { describe, it, expect, vi } from 'vitest'
import { screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { renderWithoutProviders as render } from '../../../test/utils/test-utils'
import Button from '../Button'

describe('Button', () => {
  it('renders with default props', () => {
    render(<Button>Click me</Button>)
    
    const button = screen.getByRole('button', { name: 'Click me' })
    expect(button).toBeInTheDocument()
    expect(button).toHaveClass('inline-flex', 'items-center', 'justify-center')
  })

  it('applies primary variant by default', () => {
    render(<Button>Primary</Button>)
    
    const button = screen.getByRole('button')
    expect(button).toHaveClass('bg-card', 'border-border-custom')
  })

  it('applies different variants correctly', () => {
    const { rerender } = render(<Button variant="secondary">Secondary</Button>)
    expect(screen.getByRole('button')).toHaveClass('bg-bg-secondary')

    rerender(<Button variant="danger">Danger</Button>)
    expect(screen.getByRole('button')).toHaveClass('bg-red-600')

    rerender(<Button variant="success">Success</Button>)
    expect(screen.getByRole('button')).toHaveClass('bg-green-600')

    rerender(<Button variant="warning">Warning</Button>)
    expect(screen.getByRole('button')).toHaveClass('bg-yellow-600')

    rerender(<Button variant="outline">Outline</Button>)
    expect(screen.getByRole('button')).toHaveClass('border-border-custom')

    rerender(<Button variant="ghost">Ghost</Button>)
    expect(screen.getByRole('button')).toHaveClass('text-text-primary')
  })

  it('applies different sizes correctly', () => {
    const { rerender } = render(<Button size="xs">Extra Small</Button>)
    expect(screen.getByRole('button')).toHaveClass('px-2', 'py-1', 'text-xs')

    rerender(<Button size="sm">Small</Button>)
    expect(screen.getByRole('button')).toHaveClass('px-3', 'py-1.5', 'text-sm')

    rerender(<Button size="default">Default</Button>)
    expect(screen.getByRole('button')).toHaveClass('px-5', 'py-3')

    rerender(<Button size="lg">Large</Button>)
    expect(screen.getByRole('button')).toHaveClass('px-6', 'py-4', 'text-lg')

    rerender(<Button size="xl">Extra Large</Button>)
    expect(screen.getByRole('button')).toHaveClass('px-8', 'py-5', 'text-xl')
  })

  it('handles disabled state', () => {
    render(<Button disabled>Disabled</Button>)
    
    const button = screen.getByRole('button')
    expect(button).toBeDisabled()
    expect(button).toHaveClass('disabled:opacity-50', 'disabled:cursor-not-allowed')
  })

  it('handles loading state', () => {
    render(<Button loading>Loading</Button>)
    
    const button = screen.getByRole('button')
    expect(button).toBeDisabled()
    expect(button).toHaveClass('cursor-wait')
    
    // Check for loading spinner
    const spinner = button.querySelector('svg.animate-spin')
    expect(spinner).toBeInTheDocument()
  })

  it('calls onClick handler when clicked', async () => {
    const handleClick = vi.fn()
    const user = userEvent.setup()
    
    render(<Button onClick={handleClick}>Click me</Button>)
    
    const button = screen.getByRole('button')
    await user.click(button)
    
    expect(handleClick).toHaveBeenCalledTimes(1)
  })

  it('does not call onClick when disabled', async () => {
    const handleClick = vi.fn()
    const user = userEvent.setup()
    
    render(<Button disabled onClick={handleClick}>Disabled</Button>)
    
    const button = screen.getByRole('button')
    await user.click(button)
    
    expect(handleClick).not.toHaveBeenCalled()
  })

  it('does not call onClick when loading', async () => {
    const handleClick = vi.fn()
    const user = userEvent.setup()
    
    render(<Button loading onClick={handleClick}>Loading</Button>)
    
    const button = screen.getByRole('button')
    await user.click(button)
    
    expect(handleClick).not.toHaveBeenCalled()
  })

  it('applies custom className', () => {
    render(<Button className="custom-class">Custom</Button>)
    
    const button = screen.getByRole('button')
    expect(button).toHaveClass('custom-class')
  })

  it('forwards ref correctly', () => {
    const ref = { current: null }
    
    render(<Button ref={ref}>With Ref</Button>)
    
    expect(ref.current).toBeInstanceOf(HTMLButtonElement)
  })

  it('passes through additional props', () => {
    render(<Button data-testid="test-button" aria-label="Test button">Test</Button>)
    
    const button = screen.getByTestId('test-button')
    expect(button).toHaveAttribute('aria-label', 'Test button')
  })

  it('displays loading spinner with correct styles', () => {
    render(<Button loading>Loading Button</Button>)
    
    const spinner = screen.getByRole('button').querySelector('svg')
    expect(spinner).toHaveClass('animate-spin', '-ml-1', 'mr-3', 'h-5', 'w-5')
    
    const circle = spinner?.querySelector('circle')
    const path = spinner?.querySelector('path')
    
    expect(circle).toBeInTheDocument()
    expect(path).toBeInTheDocument()
    expect(circle).toHaveClass('opacity-25')
    expect(path).toHaveClass('opacity-75')
  })

  it('maintains button text when loading', () => {
    render(<Button loading>Loading Text</Button>)
    
    expect(screen.getByText('Loading Text')).toBeInTheDocument()
  })

  it('has proper focus styles', () => {
    render(<Button>Focus me</Button>)
    
    const button = screen.getByRole('button')
    expect(button).toHaveClass('focus:outline-none', 'focus:ring-2')
  })

  it('handles keyboard interaction', async () => {
    const handleClick = vi.fn()
    const user = userEvent.setup()
    
    render(<Button onClick={handleClick}>Keyboard</Button>)
    
    const button = screen.getByRole('button')
    await user.keyboard('[Tab]')
    expect(button).toHaveFocus()
    
    await user.keyboard('[Space]')
    expect(handleClick).toHaveBeenCalledTimes(1)
    
    await user.keyboard('[Enter]')
    expect(handleClick).toHaveBeenCalledTimes(2)
  })
})
