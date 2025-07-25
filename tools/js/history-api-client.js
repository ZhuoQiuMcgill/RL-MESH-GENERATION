/**
 * History Record Dedicated API Client
 * Extends base ApiClient, specifically handles training history related API calls
 */

import {ApiClient} from './api-client.js';
import {CONSTANTS} from './utils.js';

export class HistoryApiClient extends ApiClient {
    constructor() {
        super();
        this.historyBasePath = '/training/history';
    }

    /**
     * Get training history list
     * @returns {Promise<Object>} Training history list response
     */
    async getTrainingHistoryList() {
        const fetchList = async () => {
            const response = await this.request(`${this.historyBasePath}/list`);
            return {
                success: response.success || false,
                training_ids: response.training_ids || [],
                count: response.count || 0,
                error: response.error || null
            };
        };

        try {
            return await fetchList();
        } catch (error) {
            if (error.message && error.message.toLowerCase().includes('timed out')) {
                try {
                    // Retry once if the first request timed out
                    return await fetchList();
                } catch (secondError) {
                    return {
                        success: false,
                        training_ids: [],
                        count: 0,
                        error: secondError.message
                    };
                }
            }

            return {
                success: false,
                training_ids: [],
                count: 0,
                error: error.message
            };
        }
    }

    /**
     * Get basic information of specified training
     * @param {string} trainingId - Training session ID
     * @returns {Promise<Object>} Training information response
     */
    async getTrainingInfo(trainingId) {
        try {
            const response = await this.request(`${this.historyBasePath}/info/${trainingId}`, {
                method: 'POST'
            }, CONSTANTS.HISTORY_CONNECTION_TIMEOUT);

            return {
                success: response.success || false,
                training_id: response.training_id || trainingId,
                detail_length: response.detail_length || 0,
                best_episode: response.best_episode || 0,
                error: response.error || null
            };
        } catch (error) {
            return {
                success: false,
                training_id: trainingId,
                detail_length: 0,
                best_episode: 0,
                error: error.message
            };
        }
    }

    /**
     * Get detailed data of specified Episode
     * @param {string} trainingId - Training session ID
     * @param {number} episodeIndex - Episode index
     * @returns {Promise<Object>} Episode data response
     */
    async getEpisodeData(trainingId, episodeIndex) {
        try {
            const response = await this.request(
                `${this.historyBasePath}/episode/${trainingId}/${episodeIndex}`,
                {
                    method: 'POST'
                },
                CONSTANTS.HISTORY_CONNECTION_TIMEOUT
            );

            return {
                success: response.success || false,
                training_id: response.training_id || trainingId,
                episode_index: response.episode_index || episodeIndex,
                episode_data: response.episode_data || null,
                error: response.error || null
            };
        } catch (error) {
            return {
                success: false,
                training_id: trainingId,
                episode_index: episodeIndex,
                episode_data: null,
                error: error.message
            };
        }
    }

    /**
     * Batch get Episode data
     * @param {string} trainingId - Training session ID
     * @param {Array<number>} episodeIndices - Episode index array
     * @returns {Promise<Array>} Episode data array
     */
    async getBatchEpisodeData(trainingId, episodeIndices) {
        const promises = episodeIndices.map(index =>
            this.getEpisodeData(trainingId, index)
        );

        try {
            const results = await Promise.allSettled(promises);
            return results.map((result, index) => {
                if (result.status === 'fulfilled') {
                    return result.value;
                } else {
                    return {
                        success: false,
                        training_id: trainingId,
                        episode_index: episodeIndices[index],
                        episode_data: null,
                        error: result.reason?.message || 'Unknown error'
                    };
                }
            });
        } catch (error) {
            return episodeIndices.map(index => ({
                success: false,
                training_id: trainingId,
                episode_index: index,
                episode_data: null,
                error: error.message
            }));
        }
    }

    /**
     * Get Episode range data
     * @param {string} trainingId - Training session ID
     * @param {number} startIndex - Start index
     * @param {number} endIndex - End index (inclusive)
     * @returns {Promise<Array>} Episode data array
     */
    async getEpisodeRange(trainingId, startIndex, endIndex) {
        const indices = [];
        for (let i = startIndex; i <= endIndex; i++) {
            indices.push(i);
        }
        return this.getBatchEpisodeData(trainingId, indices);
    }

    /**
     * Search Episodes (filter by reward value)
     * @param {string} trainingId - Training session ID
     * @param {number} minReward - Minimum reward value
     * @param {number} maxReward - Maximum reward value
     * @param {number} maxResults - Maximum number of results
     * @returns {Promise<Array>} Episode data array that meets criteria
     */
    async searchEpisodesByReward(trainingId, minReward = -Infinity, maxReward = Infinity, maxResults = 100) {
        try {
            // First get training information
            const trainingInfo = await this.getTrainingInfo(trainingId);
            if (!trainingInfo.success) {
                throw new Error(trainingInfo.error);
            }

            const totalEpisodes = trainingInfo.detail_length;
            const results = [];

            // Get data in batches to avoid too many concurrent requests
            const batchSize = 20;
            for (let i = 0; i < totalEpisodes && results.length < maxResults; i += batchSize) {
                const endIndex = Math.min(i + batchSize - 1, totalEpisodes - 1);
                const batchData = await this.getEpisodeRange(trainingId, i, endIndex);

                const filteredBatch = batchData.filter(item => {
                    if (!item.success || !item.episode_data) return false;
                    const reward = item.episode_data.r;
                    return reward >= minReward && reward <= maxReward;
                });

                results.push(...filteredBatch);

                if (results.length >= maxResults) {
                    results.splice(maxResults);
                    break;
                }
            }

            return results;
        } catch (error) {
            return [];
        }
    }

    /**
     * Get best Episodes
     * @param {string} trainingId - Training session ID
     * @param {number} topN - Get top N best Episodes
     * @returns {Promise<Array>} Best Episode data array
     */
    async getTopEpisodes(trainingId, topN = 10) {
        try {
            const trainingInfo = await this.getTrainingInfo(trainingId);
            if (!trainingInfo.success) {
                throw new Error(trainingInfo.error);
            }

            const totalEpisodes = trainingInfo.detail_length;
            const allEpisodes = [];

            // Get all Episode data in batches
            const batchSize = 50;
            for (let i = 0; i < totalEpisodes; i += batchSize) {
                const endIndex = Math.min(i + batchSize - 1, totalEpisodes - 1);
                const batchData = await this.getEpisodeRange(trainingId, i, endIndex);

                const validBatch = batchData.filter(item =>
                    item.success && item.episode_data && typeof item.episode_data.r === 'number'
                );

                allEpisodes.push(...validBatch);
            }

            // Sort by reward value and get top N
            allEpisodes.sort((a, b) => b.episode_data.r - a.episode_data.r);
            return allEpisodes.slice(0, topN);

        } catch (error) {
            return [];
        }
    }

    /**
     * Get training statistics
     * @param {string} trainingId - Training session ID
     * @returns {Promise<Object>} Training statistics information
     */
    async getTrainingStatistics(trainingId) {
        try {
            const trainingInfo = await this.getTrainingInfo(trainingId);
            if (!trainingInfo.success) {
                return {
                    success: false,
                    error: trainingInfo.error
                };
            }

            // Get key Episode data for statistics
            const keyEpisodes = [];
            const totalEpisodes = trainingInfo.detail_length;
            const sampleSize = Math.min(100, totalEpisodes); // Sample 100 Episodes for statistics

            // Uniform sampling
            const step = Math.floor(totalEpisodes / sampleSize);
            const sampleIndices = [];
            for (let i = 0; i < totalEpisodes; i += step) {
                sampleIndices.push(i);
            }

            // Ensure including best Episode and last Episode
            if (!sampleIndices.includes(trainingInfo.best_episode)) {
                sampleIndices.push(trainingInfo.best_episode);
            }
            if (!sampleIndices.includes(totalEpisodes - 1)) {
                sampleIndices.push(totalEpisodes - 1);
            }

            const sampleData = await this.getBatchEpisodeData(trainingId, sampleIndices);
            const validSamples = sampleData.filter(item => item.success && item.episode_data);

            if (validSamples.length === 0) {
                return {
                    success: false,
                    error: 'No valid episode data found'
                };
            }

            // Calculate statistics
            const rewards = validSamples.map(item => item.episode_data.r);
            const lengths = validSamples.map(item => item.episode_data.l);
            const completedCount = validSamples.filter(item => item.episode_data.is_completed).length;

            const stats = {
                success: true,
                training_id: trainingId,
                total_episodes: totalEpisodes,
                best_episode: trainingInfo.best_episode,
                sample_size: validSamples.length,
                reward_stats: {
                    min: Math.min(...rewards),
                    max: Math.max(...rewards),
                    mean: rewards.reduce((a, b) => a + b, 0) / rewards.length,
                    median: this.calculateMedian(rewards)
                },
                length_stats: {
                    min: Math.min(...lengths),
                    max: Math.max(...lengths),
                    mean: lengths.reduce((a, b) => a + b, 0) / lengths.length,
                    median: this.calculateMedian(lengths)
                },
                completion_rate: completedCount / validSamples.length,
                sample_episodes: validSamples.map(item => ({
                    index: item.episode_index,
                    reward: item.episode_data.r,
                    length: item.episode_data.l,
                    completed: item.episode_data.is_completed
                }))
            };

            return stats;

        } catch (error) {
            return {
                success: false,
                error: error.message
            };
        }
    }

    /**
     * Check history service health status
     * @returns {Promise<Object>} Health status response
     */
    async checkHistoryHealth() {
        try {
            const response = await this.request(`${this.historyBasePath}/health`);
            return {
                success: true,
                status: response.status || 'unknown',
                service: response.service || 'training-history-api',
                available_trainings: response.available_trainings || 0,
                current_focus: response.current_focus || null,
                timestamp: response.timestamp || Date.now()
            };
        } catch (error) {
            return {
                success: false,
                status: 'unhealthy',
                error: error.message,
                timestamp: Date.now()
            };
        }
    }

    /**
     * Validate if training ID exists
     * @param {string} trainingId - Training session ID
     * @returns {Promise<boolean>} Whether it exists
     */
    async validateTrainingId(trainingId) {
        try {
            const response = await this.getTrainingInfo(trainingId);
            return response.success;
        } catch (error) {
            return false;
        }
    }

    /**
     * Get metadata summary of training sessions
     * @param {Array<string>} trainingIds - Training ID list
     * @returns {Promise<Array>} Metadata summary array
     */
    async getTrainingsSummary(trainingIds) {
        const promises = trainingIds.map(async (trainingId) => {
            try {
                const info = await this.getTrainingInfo(trainingId);
                return {
                    training_id: trainingId,
                    success: info.success,
                    detail_length: info.detail_length,
                    best_episode: info.best_episode,
                    display_name: this.formatTrainingDisplayName(trainingId),
                    error: info.error
                };
            } catch (error) {
                return {
                    training_id: trainingId,
                    success: false,
                    detail_length: 0,
                    best_episode: 0,
                    display_name: {name: trainingId, timestamp: '', mesh: 'Unknown'},
                    error: error.message
                };
            }
        });

        try {
            const results = await Promise.allSettled(promises);
            return results.map(result =>
                result.status === 'fulfilled' ? result.value : {
                    training_id: 'unknown',
                    success: false,
                    error: result.reason?.message || 'Unknown error'
                }
            );
        } catch (error) {
            return [];
        }
    }

    /**
     * Utility method: Calculate median
     * @param {Array<number>} values - Number array
     * @returns {number} Median
     */
    calculateMedian(values) {
        if (values.length === 0) return 0;

        const sorted = [...values].sort((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2);

        if (sorted.length % 2 === 0) {
            return (sorted[mid - 1] + sorted[mid]) / 2;
        } else {
            return sorted[mid];
        }
    }

    /**
     * Utility method: Format training display name
     * @param {string} trainingId - Training ID
     * @returns {Object} Formatted display information
     */
    formatTrainingDisplayName(trainingId) {
        const parts = trainingId.split('_');
        let algorithm = 'SAC';
        let timestamp = '';
        let mesh = '';

        try {
            if (parts.length >= 4) {
                if (parts[0] === 'continue') {
                    // Continue format: continue_checkpointName_date_time_meshName
                    algorithm = 'Continue';
                    if (parts.length >= 5) {
                        // Find date-time part (8-digit_6-digit pattern)
                        let dateTimeFound = false;
                        for (let i = 1; i < parts.length - 1; i++) {
                            if (parts[i].length === 8 && /^\d{8}$/.test(parts[i]) &&
                                i + 1 < parts.length && parts[i + 1].length === 6 && /^\d{6}$/.test(parts[i + 1])) {
                                // Found date-time part
                                timestamp = `${parts[i]}_${parts[i + 1]}`;
                                // mesh name is the remaining part
                                const beforeDateTime = parts.slice(1, i);
                                const afterDateTime = parts.slice(i + 2);
                                mesh = [...afterDateTime, ...beforeDateTime].join('_');
                                dateTimeFound = true;
                                break;
                            }
                        }

                        if (!dateTimeFound) {
                            // Fall back to old logic
                            timestamp = parts.length >= 4 ? `${parts[2]}_${parts[3]}` : '';
                            mesh = parts.slice(4).join('_');
                        }
                    } else {
                        // Less than 5 parts, use remaining as mesh name
                        mesh = parts.slice(1).join('_');
                        timestamp = '';
                    }
                } else {
                    // Normal format: algorithm_date_time_meshName
                    algorithm = parts[0].toUpperCase();
                    if (parts.length >= 3 && parts[1].length === 8 && /^\d{8}$/.test(parts[1]) &&
                        parts[2].length === 6 && /^\d{6}$/.test(parts[2])) {
                        timestamp = `${parts[1]}_${parts[2]}`;
                        mesh = parts.slice(3).join('_');
                    } else {
                        // Fallback handling
                        timestamp = parts.slice(1, 3).join('_');
                        mesh = parts.slice(3).join('_');
                    }
                }
            } else {
                // Less than 4 parts, cannot parse correctly
                algorithm = 'Unknown';
                timestamp = '';
                mesh = trainingId;
            }
        } catch (error) {
            // Parsing error, use original values
            console.warn('Training ID parsing failed:', trainingId, error);
            algorithm = 'Unknown';
            timestamp = '';
            mesh = trainingId;
        }

        // Format timestamp display
        let formattedTimestamp = timestamp;
        if (timestamp && timestamp.includes('_')) {
            const [date, time] = timestamp.split('_');
            if (date.length === 8 && time.length === 6) {
                const formattedDate = `${date.slice(0, 4)}-${date.slice(4, 6)}-${date.slice(6, 8)}`;
                const formattedTime = `${time.slice(0, 2)}:${time.slice(2, 4)}:${time.slice(4, 6)}`;
                formattedTimestamp = `${formattedDate} ${formattedTime}`;
            }
        }

        return {
            name: `${algorithm} - ${mesh || 'Unknown'}`,
            timestamp: formattedTimestamp,
            mesh: mesh || 'Unknown',
            algorithm: algorithm
        };
    }

    /**
     * Export Episode data to CSV format
     * @param {string} trainingId - Training session ID
     * @param {Array<number>} episodeIndices - Episode index array (optional, default export all)
     * @returns {Promise<string>} CSV format data
     */
    async exportEpisodesToCSV(trainingId, episodeIndices = null) {
        try {
            let episodeData;

            if (episodeIndices) {
                episodeData = await this.getBatchEpisodeData(trainingId, episodeIndices);
            } else {
                // Export all Episodes
                const trainingInfo = await this.getTrainingInfo(trainingId);
                if (!trainingInfo.success) {
                    throw new Error(trainingInfo.error);
                }

                const allIndices = Array.from({length: trainingInfo.detail_length}, (_, i) => i);
                episodeData = await this.getBatchEpisodeData(trainingId, allIndices);
            }

            const validData = episodeData.filter(item => item.success && item.episode_data);

            if (validData.length === 0) {
                throw new Error('No valid episode data to export');
            }

            // Generate CSV header
            const headers = [
                'episode_index',
                'reward',
                'length',
                'completed',
                'boundary_vertices_count',
                'mesh_vertices_count'
            ];

            // Generate CSV content
            const csvRows = [headers.join(',')];

            validData.forEach(item => {
                const data = item.episode_data;
                const row = [
                    item.episode_index,
                    data.r,
                    data.l,
                    data.is_completed ? 1 : 0,
                    (data.boundary_vertices_data || []).length,
                    Object.keys(data.mesh_data || {}).length
                ];
                csvRows.push(row.join(','));
            });

            return csvRows.join('\n');

        } catch (error) {
            throw new Error(`CSV export failed: ${error.message}`);
        }
    }
}