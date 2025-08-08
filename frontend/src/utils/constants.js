/**
 * Constants and utility functions
 * Consolidated utilities for the application
 */

// Main application constants
export const CONSTANTS = {
    API_BASE_URL: 'http://127.0.0.1:5000',
    VERTEX_RADIUS: 3,
    GRID_SIZE: 20,
    DEFAULT_PADDING: 40,
    CANVAS_DEVICE_PIXEL_RATIO: typeof window !== 'undefined' ? (window.devicePixelRatio || 1) : 1,
    CONNECTION_TIMEOUT: 60000,
    TRAINING_STOP_TIMEOUT: 30000,
};

// Status constants
export const STATUS = {
    RUNNING: 'running',
    STOPPED: 'stopped',
    COMPLETED: 'completed',
    STOPPING: 'stopping',
    ERROR: 'error',
    IDLE: 'idle'
};

/**
 * Utility function for merging class names conditionally
 * @param {...string} classes - Class names to merge
 * @returns {string} Merged class names
 */
export function cn(...classes) {
    return classes.filter(Boolean).join(' ');
}

/**
 * Check if coordinate is valid
 * @param {Array} coord - Coordinate array [x, y]
 * @returns {boolean} True if valid
 */
export function isValidCoordinate(coord) {
    return Array.isArray(coord) && 
           coord.length >= 2 && 
           typeof coord[0] === 'number' && 
           typeof coord[1] === 'number' &&
           !isNaN(coord[0]) && 
           !isNaN(coord[1]) &&
           isFinite(coord[0]) && 
           isFinite(coord[1]);
}

/**
 * Parse backend data safely
 * @param {*} data - Data to parse
 * @returns {*} Parsed data
 */
export function parseBackendData(data) {
    if (typeof data === 'string') {
        try {
            return JSON.parse(data);
        } catch (e) {
            console.warn('Failed to parse JSON data:', e);
            return data;
        }
    }
    return data;
}

/**
 * Format number display
 * @param {number} num - The number to format
 * @param {number} decimals - Number of decimal places, default 3
 * @returns {string} Formatted string
 */
export function formatNumber(num, decimals = 3) {
    return (num !== undefined && num !== null) ? num.toFixed(decimals) : 'N/A';
}

/**
 * Create delay function - Promise-based for async/await support
 * @param {number} ms - Delay time (milliseconds)
 * @returns {Promise} Promise object
 */
export function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}
