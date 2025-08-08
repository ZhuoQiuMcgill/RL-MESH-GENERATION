import { test, expect } from '@playwright/test'

test.describe('Dashboard Smoke Tests', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to dashboard
    await page.goto('/')
  })

  test('should load the dashboard successfully', async ({ page }) => {
    // Check that the page loads and displays the main title
    await expect(page.getByText('RL Mesh Generation')).toBeVisible()
    await expect(page.getByText('Explore our reinforcement learning tools')).toBeVisible()
  })

  test('should display all module cards', async ({ page }) => {
    // Wait for the page to load completely
    await page.waitForLoadState('networkidle')

    // Check that all expected modules are visible
    const modules = [
      'Training',
      'History', 
      'Quality Analysis',
      'Geometry Tools',
      'Canvas',
      'Angle Analysis',
      'Action Spaces',
      'Generator'
    ]

    for (const module of modules) {
      await expect(page.getByText(module)).toBeVisible()
    }
  })

  test('should display statistics cards', async ({ page }) => {
    // Check statistics section
    await expect(page.getByText('Overview')).toBeVisible()
    await expect(page.getByText('Training Episodes')).toBeVisible()
    await expect(page.getByText('Quality Score')).toBeVisible()
    await expect(page.getByText('Generated Meshes')).toBeVisible()
    await expect(page.getByText('Active Models')).toBeVisible()
  })

  test('should navigate to training page', async ({ page }) => {
    // Click on the Training module card
    await page.getByRole('link', { name: /Training/ }).click()
    
    // Should navigate to /train
    await expect(page).toHaveURL('/train')
  })

  test('should navigate to history page', async ({ page }) => {
    // Click on the History module card
    await page.getByRole('link', { name: /History/ }).click()
    
    // Should navigate to /history
    await expect(page).toHaveURL('/history')
  })

  test('should navigate to quality analysis page', async ({ page }) => {
    // Click on the Quality Analysis module card
    await page.getByRole('link', { name: /Quality Analysis/ }).click()
    
    // Should navigate to /quality
    await expect(page).toHaveURL('/quality')
  })

  test('should navigate to canvas page', async ({ page }) => {
    // Click on the Canvas module card
    await page.getByRole('link', { name: /Canvas/ }).click()
    
    // Should navigate to /canvas
    await expect(page).toHaveURL('/canvas')
  })

  test('should have responsive layout on mobile', async ({ page }) => {
    // Set viewport to mobile size
    await page.setViewportSize({ width: 375, height: 667 })
    
    // Reload to apply mobile styles
    await page.reload()
    await page.waitForLoadState('networkidle')
    
    // Check that content is still visible and properly laid out
    await expect(page.getByText('RL Mesh Generation')).toBeVisible()
    await expect(page.getByText('Training')).toBeVisible()
    await expect(page.getByText('Overview')).toBeVisible()
  })

  test('should display recent activity section', async ({ page }) => {
    await expect(page.getByText('Recent Activity')).toBeVisible()
    await expect(page.getByText('Training session completed')).toBeVisible()
    await expect(page.getByText('New mesh generated')).toBeVisible()
    await expect(page.getByText('Analysis completed')).toBeVisible()
  })

  test('should have accessible module links', async ({ page }) => {
    // Check that module links are properly accessible
    const trainingLink = page.getByRole('link', { name: /Training/ })
    await expect(trainingLink).toBeVisible()
    await expect(trainingLink).toHaveAttribute('href', '/train')
    
    const historyLink = page.getByRole('link', { name: /History/ })
    await expect(historyLink).toBeVisible()
    await expect(historyLink).toHaveAttribute('href', '/history')
  })

  test('should handle keyboard navigation', async ({ page }) => {
    // Test keyboard navigation through module links
    await page.keyboard.press('Tab')
    await page.keyboard.press('Tab')
    await page.keyboard.press('Tab')
    
    // Should be able to navigate with keyboard
    const focusedElement = await page.locator(':focus')
    await expect(focusedElement).toBeVisible()
  })
})
