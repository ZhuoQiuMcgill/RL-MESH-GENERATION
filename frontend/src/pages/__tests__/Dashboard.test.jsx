import { describe, it, expect } from 'vitest'
import { screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { renderWithRouter } from '../../test/utils/test-utils'
import Dashboard from '../Dashboard'

describe('Dashboard', () => {
  it('renders dashboard title and subtitle', () => {
    renderWithRouter(<Dashboard />)
    
    expect(screen.getByText('RL Mesh Generation')).toBeInTheDocument()
    expect(screen.getByText(/Explore our reinforcement learning tools/)).toBeInTheDocument()
  })

  it('renders overview section', () => {
    renderWithRouter(<Dashboard />)
    
    expect(screen.getByText('Overview')).toBeInTheDocument()
  })

  it('renders statistics cards', () => {
    renderWithRouter(<Dashboard />)
    
    expect(screen.getByText('Training Episodes')).toBeInTheDocument()
    expect(screen.getByText('156')).toBeInTheDocument()
    expect(screen.getByText('Total completed episodes')).toBeInTheDocument()
    
    expect(screen.getByText('Quality Score')).toBeInTheDocument()
    expect(screen.getByText('89.2%')).toBeInTheDocument()
    expect(screen.getByText('Average mesh quality')).toBeInTheDocument()
    
    expect(screen.getByText('Generated Meshes')).toBeInTheDocument()
    expect(screen.getByText('1,247')).toBeInTheDocument()
    expect(screen.getByText('Total meshes created')).toBeInTheDocument()
    
    expect(screen.getByText('Active Models')).toBeInTheDocument()
    expect(screen.getByText('4')).toBeInTheDocument()
    expect(screen.getByText('Models ready for training')).toBeInTheDocument()
  })

  it('renders modules section', () => {
    renderWithRouter(<Dashboard />)
    
    expect(screen.getByText('Modules')).toBeInTheDocument()
  })

  it('renders all module cards', () => {
    renderWithRouter(<Dashboard />)
    
    const expectedModules = [
      'Training',
      'History',
      'Quality Analysis',
      'Geometry Tools',
      'Canvas',
      'Angle Analysis',
      'Action Spaces',
      'Generator'
    ]

    expectedModules.forEach(module => {
      expect(screen.getByText(module)).toBeInTheDocument()
    })
  })

  it('renders module descriptions', () => {
    renderWithRouter(<Dashboard />)
    
    expect(screen.getByText('Start or monitor training sessions')).toBeInTheDocument()
    expect(screen.getByText('View training history and logs')).toBeInTheDocument()
    expect(screen.getByText('Analyze mesh quality metrics')).toBeInTheDocument()
    expect(screen.getByText('Geometry manipulation tools')).toBeInTheDocument()
    expect(screen.getByText('Interactive 3D mesh canvas')).toBeInTheDocument()
    expect(screen.getByText('Analyze mesh angles and topology')).toBeInTheDocument()
    expect(screen.getByText('Configure RL action spaces')).toBeInTheDocument()
    expect(screen.getByText('Mesh generation tools')).toBeInTheDocument()
  })

  it('renders module cards as links', () => {
    renderWithRouter(<Dashboard />)
    
    const trainingLink = screen.getByRole('link', { name: /Training/ })
    expect(trainingLink).toHaveAttribute('href', '/train')
    
    const historyLink = screen.getByRole('link', { name: /History/ })
    expect(historyLink).toHaveAttribute('href', '/history')
    
    const qualityLink = screen.getByRole('link', { name: /Quality Analysis/ })
    expect(qualityLink).toHaveAttribute('href', '/quality')
    
    const geometryLink = screen.getByRole('link', { name: /Geometry Tools/ })
    expect(geometryLink).toHaveAttribute('href', '/geometry')
    
    const canvasLink = screen.getByRole('link', { name: /Canvas/ })
    expect(canvasLink).toHaveAttribute('href', '/canvas')
    
    const angleLink = screen.getByRole('link', { name: /Angle Analysis/ })
    expect(angleLink).toHaveAttribute('href', '/angle')
    
    const actionLink = screen.getByRole('link', { name: /Action Spaces/ })
    expect(actionLink).toHaveAttribute('href', '/action')
    
    const generatorLink = screen.getByRole('link', { name: /Generator/ })
    expect(generatorLink).toHaveAttribute('href', '/generator')
  })

  it('renders module icons', () => {
    renderWithRouter(<Dashboard />)
    
    expect(screen.getByText('🚂')).toBeInTheDocument() // Training
    expect(screen.getByText('📋')).toBeInTheDocument() // History
    expect(screen.getByText('⭐')).toBeInTheDocument() // Quality
    expect(screen.getByText('📐')).toBeInTheDocument() // Geometry
    expect(screen.getByText('🎨')).toBeInTheDocument() // Canvas
    expect(screen.getByText('📊')).toBeInTheDocument() // Angle
    expect(screen.getByText('⚡')).toBeInTheDocument() // Action
    expect(screen.getByText('🔧')).toBeInTheDocument() // Generator
  })

  it('renders recent activity section', () => {
    renderWithRouter(<Dashboard />)
    
    expect(screen.getByText('Recent Activity')).toBeInTheDocument()
    expect(screen.getByText('Training session completed')).toBeInTheDocument()
    expect(screen.getByText('PPO model on simple_square mesh')).toBeInTheDocument()
    expect(screen.getByText('2 hours ago')).toBeInTheDocument()
    
    expect(screen.getByText('New mesh generated')).toBeInTheDocument()
    expect(screen.getByText('Quality score: 92.4%')).toBeInTheDocument()
    expect(screen.getByText('4 hours ago')).toBeInTheDocument()
    
    expect(screen.getByText('Analysis completed')).toBeInTheDocument()
    expect(screen.getByText('Angle distribution report generated')).toBeInTheDocument()
    expect(screen.getByText('6 hours ago')).toBeInTheDocument()
  })

  it('has hover effects on module cards', async () => {
    const user = userEvent.setup()
    renderWithRouter(<Dashboard />)
    
    const trainingCard = screen.getByRole('link', { name: /Training/ }).closest('div')
    
    await user.hover(trainingCard)
    
    // Check for hover classes - the component should have hover:shadow-lg and hover:scale-[1.02]
    expect(trainingCard).toHaveClass('hover:shadow-lg', 'transition-all', 'duration-200', 'hover:scale-[1.02]')
  })

  it('shows explore text on all module cards', () => {
    renderWithRouter(<Dashboard />)
    
    const exploreTexts = screen.getAllByText('Explore')
    expect(exploreTexts).toHaveLength(8) // One for each module
  })

  it('renders statistics with trend indicators', () => {
    renderWithRouter(<Dashboard />)
    
    expect(screen.getByText('+12%')).toBeInTheDocument()
    expect(screen.getByText('+2.1%')).toBeInTheDocument()
    expect(screen.getByText('+23')).toBeInTheDocument()
  })

  it('renders activity icons', () => {
    renderWithRouter(<Dashboard />)
    
    const activityItems = screen.getByText('Recent Activity').closest('div')
    
    // Check that activity items have their colored background indicators
    expect(activityItems.querySelector('.bg-blue-500\\/10')).toBeInTheDocument()
    expect(activityItems.querySelector('.bg-green-500\\/10')).toBeInTheDocument()
    expect(activityItems.querySelector('.bg-yellow-500\\/10')).toBeInTheDocument()
  })

  it('has correct layout structure', () => {
    renderWithRouter(<Dashboard />)
    
    // Check main container
    const mainContainer = screen.getByText('RL Mesh Generation').closest('div')
    expect(mainContainer).toHaveClass('max-w-7xl', 'mx-auto', 'p-6')
    
    // Check grid layouts
    const statsGrid = screen.getByText('Training Episodes').closest('div').closest('div')
    expect(statsGrid).toHaveClass('grid', 'grid-cols-1', 'md:grid-cols-2', 'lg:grid-cols-4', 'gap-6')
    
    const modulesGrid = screen.getByText('Training').closest('div').closest('div').closest('div')
    expect(modulesGrid).toHaveClass('grid', 'grid-cols-1', 'md:grid-cols-2', 'lg:grid-cols-3', 'xl:grid-cols-4', 'gap-6')
  })
})
