/**
 * Utility function to merge class names
 * Simple implementation for className concatenation and conditional classes
 * @param {...(string|boolean|undefined)} classes - Class names to merge
 * @returns {string} Merged class names
 */
export function cn(...classes) {
  return classes
    .filter(Boolean)
    .join(' ')
    .trim();
}

export default cn;
