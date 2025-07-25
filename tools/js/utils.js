/**
 * Utility Functions Module
 * Contains constant definitions, formatting functions, and common utility functions
 */

// Constant definitions
export const CONSTANTS = {
    API_BASE_URL: 'http://127.0.0.1:5000',
    CANVAS_DEVICE_PIXEL_RATIO: window.devicePixelRatio || 1,
    GRID_SIZE: 20,
    DEFAULT_PADDING: 50,
    CANVAS_PADDING: 32,
    MAX_LOGS: 100,
    CONNECTION_TIMEOUT: 60000, // Increased to 60 seconds
    HISTORY_CONNECTION_TIMEOUT: 120000, // History records timeout: 2 minutes
    TRAINING_STOP_TIMEOUT: 30000, // Training stop timeout: 30 seconds
    VERTEX_RADIUS: 6,
    UPDATE_INTERVALS: {
        DEFAULT: 10000 // 10 seconds
    }
};

// Status-related constants
export const STATUS = {
    RUNNING: 'running',
    STOPPED: 'stopped',
    COMPLETED: 'completed',
    STOPPING: 'stopping',
    ERROR: 'error',
    IDLE: 'idle'
};

// Log types
export const LOG_TYPES = {
    SUCCESS: 'success',
    ERROR: 'error',
    WARNING: 'warning',
    INFO: 'info'
};

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
 * Get current timestamp string
 * @returns {string} Timestamp string
 */
export function getTimestamp() {
    return new Date().toLocaleTimeString();
}

/**
 * Deep clean data to ensure JSON safety
 * @param {any} data - Data to clean
 * @returns {any} Cleaned data
 */
export function deepCleanForJSON(data) {
    if (data === null || data === undefined) {
        return null;
    }

    if (typeof data === 'boolean' || typeof data === 'number' || typeof data === 'string') {
        return data;
    }

    if (data.constructor === Array) {
        return data.map(item => deepCleanForJSON(item));
    }

    if (data.constructor === Object) {
        const cleaned = {};
        for (const [key, value] of Object.entries(data)) {
            cleaned[String(key)] = deepCleanForJSON(value);
        }
        return cleaned;
    }

    // For other types, try to convert to string
    try {
        return String(data);
    } catch {
        return null;
    }
}

/**
 * Debounce function
 * @param {Function} func - Function to debounce
 * @param {number} wait - Wait time (milliseconds)
 * @returns {Function} Debounced function
 */
export function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Throttle function
 * @param {Function} func - Function to throttle
 * @param {number} limit - Limit time (milliseconds)
 * @returns {Function} Throttled function
 */
export function throttle(func, limit) {
    let lastFunc;
    let lastRan;
    return function (...args) {
        if (!lastRan) {
            func.apply(this, args);
            lastRan = Date.now();
        } else {
            clearTimeout(lastFunc);
            lastFunc = setTimeout(() => {
                if ((Date.now() - lastRan) >= limit) {
                    func.apply(this, args);
                    lastRan = Date.now();
                }
            }, limit - (Date.now() - lastRan));
        }
    };
}

/**
 * Create delay function
 * @param {number} ms - Delay time (milliseconds)
 * @returns {Promise} Promise object
 */
export function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Parse backend returned data
 * @param {any} data - Data to parse
 * @returns {any} Parsed data
 */
export function parseBackendData(data) {
    if (typeof data === 'string') {
        try {
            return JSON.parse(data);
        } catch (e) {
            console.error('Failed to parse data:', e);
            return null;
        }
    }
    return data;
}

/**
 * Validate coordinate data
 * @param {any} coords - Coordinate data
 * @returns {boolean} Whether it is a valid coordinate
 */
export function isValidCoordinate(coords) {
    return Array.isArray(coords) &&
        coords.length === 2 &&
        typeof coords[0] === 'number' &&
        typeof coords[1] === 'number' &&
        !isNaN(coords[0]) &&
        !isNaN(coords[1]);
}

/**
 * Calculate distance between two points
 * @param {Array} p1 - Point 1 [x, y]
 * @param {Array} p2 - Point 2 [x, y]
 * @returns {number} Distance value
 */
export function calculateDistance(p1, p2) {
    if (!isValidCoordinate(p1) || !isValidCoordinate(p2)) {
        return 0;
    }
    const dx = p1[0] - p2[0];
    const dy = p1[1] - p2[1];
    return Math.sqrt(dx * dx + dy * dy);
}

/**
 * Get log style
 * @param {string} type - Log type
 * @returns {Object} Style object
 */
export function getLogStyle(type) {
    const styles = {
        [LOG_TYPES.SUCCESS]: {color: '#059669', icon: '✓'},
        [LOG_TYPES.ERROR]: {color: '#DC2626', icon: '✗'},
        [LOG_TYPES.WARNING]: {color: '#D97706', icon: '⚠'},
        [LOG_TYPES.INFO]: {color: '#6B7280', icon: 'ℹ'}
    };

    return styles[type] || styles[LOG_TYPES.INFO];
}

/**
 * Safely get DOM element
 * @param {string} id - Element ID
 * @returns {HTMLElement|null} DOM element or null
 */
export function safeGetElement(id) {
    try {
        return document.getElementById(id);
    } catch (error) {
        console.warn(`Failed to get element ${id}:`, error);
        return null;
    }
}