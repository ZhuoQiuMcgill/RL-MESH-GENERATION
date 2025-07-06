/**
 * 工具函数模块
 * 包含常量定义、格式化函数和通用工具函数
 */

// 常量定义
export const CONSTANTS = {
    API_BASE_URL: 'http://127.0.0.1:5000',
    CANVAS_DEVICE_PIXEL_RATIO: window.devicePixelRatio || 1,
    GRID_SIZE: 20,
    DEFAULT_PADDING: 50,
    CANVAS_PADDING: 32,
    MAX_LOGS: 100,
    CONNECTION_TIMEOUT: 5000,
    VERTEX_RADIUS: 6,
    UPDATE_INTERVALS: {
        DEFAULT: 10000 // 10秒
    }
};

// 状态相关常量
export const STATUS = {
    RUNNING: 'running',
    STOPPED: 'stopped',
    COMPLETED: 'completed',
    STOPPING: 'stopping',
    ERROR: 'error',
    IDLE: 'idle'
};

// 日志类型
export const LOG_TYPES = {
    SUCCESS: 'success',
    ERROR: 'error',
    WARNING: 'warning',
    INFO: 'info'
};

/**
 * 格式化数字显示
 * @param {number} num - 要格式化的数字
 * @param {number} decimals - 小数位数，默认3位
 * @returns {string} 格式化后的字符串
 */
export function formatNumber(num, decimals = 3) {
    return (num !== undefined && num !== null) ? num.toFixed(decimals) : 'N/A';
}

/**
 * 获取当前时间戳字符串
 * @returns {string} 时间戳字符串
 */
export function getTimestamp() {
    return new Date().toLocaleTimeString();
}

/**
 * 深度清理数据，确保JSON安全
 * @param {any} data - 要清理的数据
 * @returns {any} 清理后的数据
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

    // 对于其他类型，尝试转换为字符串
    try {
        return String(data);
    } catch {
        return null;
    }
}

/**
 * 防抖函数
 * @param {Function} func - 要防抖的函数
 * @param {number} wait - 等待时间（毫秒）
 * @returns {Function} 防抖后的函数
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
 * 节流函数
 * @param {Function} func - 要节流的函数
 * @param {number} limit - 限制时间（毫秒）
 * @returns {Function} 节流后的函数
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
 * 创建延迟函数
 * @param {number} ms - 延迟时间（毫秒）
 * @returns {Promise} Promise对象
 */
export function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * 解析后端返回的数据
 * @param {any} data - 要解析的数据
 * @returns {any} 解析后的数据
 */
export function parseBackendData(data) {
    if (typeof data === 'string') {
        try {
            return JSON.parse(data);
        } catch (e) {
            console.error('解析数据失败:', e);
            return null;
        }
    }
    return data;
}

/**
 * 验证坐标数据
 * @param {any} coords - 坐标数据
 * @returns {boolean} 是否为有效坐标
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
 * 计算两点间距离
 * @param {Array} p1 - 点1 [x, y]
 * @param {Array} p2 - 点2 [x, y]
 * @returns {number} 距离值
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
 * 获取日志样式
 * @param {string} type - 日志类型
 * @returns {Object} 样式对象
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
 * 安全地获取DOM元素
 * @param {string} id - 元素ID
 * @returns {HTMLElement|null} DOM元素或null
 */
export function safeGetElement(id) {
    try {
        return document.getElementById(id);
    } catch (error) {
        console.warn(`无法获取元素 ${id}:`, error);
        return null;
    }
}