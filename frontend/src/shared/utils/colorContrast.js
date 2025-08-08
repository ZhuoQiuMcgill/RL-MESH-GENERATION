/**
 * Color Contrast Utilities for WCAG AA Compliance
 * 
 * This utility provides functions to calculate and verify color contrast ratios
 * according to WCAG (Web Content Accessibility Guidelines) standards.
 * 
 * WCAG Requirements:
 * - AA Normal Text: 4.5:1 contrast ratio minimum
 * - AA Large Text: 3:1 contrast ratio minimum
 * - AAA Normal Text: 7:1 contrast ratio minimum
 * - AAA Large Text: 4.5:1 contrast ratio minimum
 */

/**
 * Convert hex color to RGB values
 * @param {string} hex - Hex color code (with or without #)
 * @returns {Object} RGB values {r, g, b}
 */
export const hexToRgb = (hex) => {
  const cleanHex = hex.replace('#', '')
  
  if (cleanHex.length === 3) {
    return {
      r: parseInt(cleanHex[0] + cleanHex[0], 16),
      g: parseInt(cleanHex[1] + cleanHex[1], 16),
      b: parseInt(cleanHex[2] + cleanHex[2], 16)
    }
  }
  
  return {
    r: parseInt(cleanHex.substr(0, 2), 16),
    g: parseInt(cleanHex.substr(2, 2), 16),
    b: parseInt(cleanHex.substr(4, 2), 16)
  }
}

/**
 * Calculate relative luminance of a color
 * @param {Object} rgb - RGB values {r, g, b}
 * @returns {number} Relative luminance value
 */
export const getRelativeLuminance = (rgb) => {
  const { r, g, b } = rgb
  
  // Convert RGB to sRGB
  const rsRGB = r / 255
  const gsRGB = g / 255
  const bsRGB = b / 255
  
  // Apply gamma correction
  const rLinear = rsRGB <= 0.03928 ? rsRGB / 12.92 : Math.pow((rsRGB + 0.055) / 1.055, 2.4)
  const gLinear = gsRGB <= 0.03928 ? gsRGB / 12.92 : Math.pow((gsRGB + 0.055) / 1.055, 2.4)
  const bLinear = bsRGB <= 0.03928 ? bsRGB / 12.92 : Math.pow((bsRGB + 0.055) / 1.055, 2.4)
  
  // Calculate relative luminance
  return 0.2126 * rLinear + 0.7152 * gLinear + 0.0722 * bLinear
}

/**
 * Calculate contrast ratio between two colors
 * @param {string} color1 - First color (hex)
 * @param {string} color2 - Second color (hex)
 * @returns {number} Contrast ratio
 */
export const getContrastRatio = (color1, color2) => {
  const rgb1 = hexToRgb(color1)
  const rgb2 = hexToRgb(color2)
  
  const lum1 = getRelativeLuminance(rgb1)
  const lum2 = getRelativeLuminance(rgb2)
  
  const lighter = Math.max(lum1, lum2)
  const darker = Math.min(lum1, lum2)
  
  return (lighter + 0.05) / (darker + 0.05)
}

/**
 * Check if colors meet WCAG AA standards
 * @param {string} foreground - Foreground color (hex)
 * @param {string} background - Background color (hex)
 * @param {string} textSize - Text size ('normal' or 'large')
 * @returns {Object} Compliance results
 */
export const checkWCAGCompliance = (foreground, background, textSize = 'normal') => {
  const ratio = getContrastRatio(foreground, background)
  
  const requirements = {
    AA_normal: 4.5,
    AA_large: 3.0,
    AAA_normal: 7.0,
    AAA_large: 4.5
  }
  
  const isLargeText = textSize === 'large'
  const aaRequired = isLargeText ? requirements.AA_large : requirements.AA_normal
  const aaaRequired = isLargeText ? requirements.AAA_large : requirements.AAA_normal
  
  return {
    ratio: Math.round(ratio * 100) / 100,
    AA: ratio >= aaRequired,
    AAA: ratio >= aaaRequired,
    level: ratio >= aaaRequired ? 'AAA' : ratio >= aaRequired ? 'AA' : 'Fail',
    textSize,
    recommendations: ratio < aaRequired ? getRecommendations(foreground, background, ratio, aaRequired) : null
  }
}

/**
 * Get color recommendations to improve contrast
 * @param {string} foreground - Foreground color (hex)
 * @param {string} background - Background color (hex)
 * @param {number} currentRatio - Current contrast ratio
 * @param {number} targetRatio - Target contrast ratio
 * @returns {Object} Recommendations
 */
const getRecommendations = (foreground, background, currentRatio, targetRatio) => {
  return {
    message: `Current ratio ${Math.round(currentRatio * 100) / 100} does not meet WCAG AA requirement of ${targetRatio}`,
    suggestions: [
      'Darken the foreground color',
      'Lighten the background color', 
      'Use a different color combination',
      'Consider using our predefined semantic color variants'
    ]
  }
}

/**
 * Predefined color combinations that meet WCAG AA standards
 */
export const accessibleColorCombinations = {
  light: {
    // Light theme combinations
    primary: { foreground: '#1e40af', background: '#ffffff' }, // Blue on white
    secondary: { foreground: '#6b7280', background: '#ffffff' }, // Gray on white
    success: { foreground: '#059669', background: '#ffffff' }, // Green on white
    warning: { foreground: '#d97706', background: '#ffffff' }, // Orange on white
    danger: { foreground: '#dc2626', background: '#ffffff' }, // Red on white
    muted: { foreground: '#6b7280', background: '#f9fafb' }, // Gray on light gray
  },
  dark: {
    // Dark theme combinations
    primary: { foreground: '#60a5fa', background: '#1f2937' }, // Light blue on dark gray
    secondary: { foreground: '#d1d5db', background: '#1f2937' }, // Light gray on dark gray
    success: { foreground: '#34d399', background: '#1f2937' }, // Light green on dark gray
    warning: { foreground: '#fbbf24', background: '#1f2937' }, // Light yellow on dark gray
    danger: { foreground: '#f87171', background: '#1f2937' }, // Light red on dark gray
    muted: { foreground: '#9ca3af', background: '#374151' }, // Medium gray on darker gray
  }
}

/**
 * Test a set of color combinations for accessibility
 * @param {Object} combinations - Object of color combinations to test
 * @returns {Object} Test results
 */
export const testColorCombinations = (combinations) => {
  const results = {}
  
  for (const [name, colors] of Object.entries(combinations)) {
    results[name] = checkWCAGCompliance(colors.foreground, colors.background)
  }
  
  return results
}

/**
 * Validate current theme colors for accessibility
 * @returns {Object} Validation results for light and dark themes
 */
export const validateThemeColors = () => {
  return {
    light: testColorCombinations(accessibleColorCombinations.light),
    dark: testColorCombinations(accessibleColorCombinations.dark)
  }
}
