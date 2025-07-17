/**
 * 历史记录专用API客户端
 * 扩展基础ApiClient，专门处理训练历史相关的API调用
 */

import {ApiClient} from './api-client.js';

export class HistoryApiClient extends ApiClient {
    constructor() {
        super();
        this.historyBasePath = '/training/history';
    }

    /**
     * 获取训练历史列表
     * @returns {Promise<Object>} 训练历史列表响应
     */
    async getTrainingHistoryList() {
        try {
            const response = await this.request(`${this.historyBasePath}/list`);
            return {
                success: response.success || false,
                training_ids: response.training_ids || [],
                count: response.count || 0,
                error: response.error || null
            };
        } catch (error) {
            return {
                success: false,
                training_ids: [],
                count: 0,
                error: error.message
            };
        }
    }

    /**
     * 获取指定训练的基本信息
     * @param {string} trainingId - 训练会话ID
     * @returns {Promise<Object>} 训练信息响应
     */
    async getTrainingInfo(trainingId) {
        try {
            const response = await this.request(`${this.historyBasePath}/info/${trainingId}`, {
                method: 'POST'
            });

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
     * 获取指定Episode的详细数据
     * @param {string} trainingId - 训练会话ID
     * @param {number} episodeIndex - Episode索引
     * @returns {Promise<Object>} Episode数据响应
     */
    async getEpisodeData(trainingId, episodeIndex) {
        try {
            const response = await this.request(
                `${this.historyBasePath}/episode/${trainingId}/${episodeIndex}`,
                {method: 'POST'}
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
     * 批量获取Episode数据
     * @param {string} trainingId - 训练会话ID
     * @param {Array<number>} episodeIndices - Episode索引数组
     * @returns {Promise<Array>} Episode数据数组
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
     * 获取Episode范围数据
     * @param {string} trainingId - 训练会话ID
     * @param {number} startIndex - 起始索引
     * @param {number} endIndex - 结束索引（包含）
     * @returns {Promise<Array>} Episode数据数组
     */
    async getEpisodeRange(trainingId, startIndex, endIndex) {
        const indices = [];
        for (let i = startIndex; i <= endIndex; i++) {
            indices.push(i);
        }
        return this.getBatchEpisodeData(trainingId, indices);
    }

    /**
     * 搜索Episodes（按奖励值筛选）
     * @param {string} trainingId - 训练会话ID
     * @param {number} minReward - 最小奖励值
     * @param {number} maxReward - 最大奖励值
     * @param {number} maxResults - 最大结果数量
     * @returns {Promise<Array>} 符合条件的Episode数据数组
     */
    async searchEpisodesByReward(trainingId, minReward = -Infinity, maxReward = Infinity, maxResults = 100) {
        try {
            // 首先获取训练信息
            const trainingInfo = await this.getTrainingInfo(trainingId);
            if (!trainingInfo.success) {
                throw new Error(trainingInfo.error);
            }

            const totalEpisodes = trainingInfo.detail_length;
            const results = [];

            // 分批获取数据以避免过多并发请求
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
     * 获取最佳Episodes
     * @param {string} trainingId - 训练会话ID
     * @param {number} topN - 获取前N个最佳Episode
     * @returns {Promise<Array>} 最佳Episode数据数组
     */
    async getTopEpisodes(trainingId, topN = 10) {
        try {
            const trainingInfo = await this.getTrainingInfo(trainingId);
            if (!trainingInfo.success) {
                throw new Error(trainingInfo.error);
            }

            const totalEpisodes = trainingInfo.detail_length;
            const allEpisodes = [];

            // 分批获取所有Episode数据
            const batchSize = 50;
            for (let i = 0; i < totalEpisodes; i += batchSize) {
                const endIndex = Math.min(i + batchSize - 1, totalEpisodes - 1);
                const batchData = await this.getEpisodeRange(trainingId, i, endIndex);

                const validBatch = batchData.filter(item =>
                    item.success && item.episode_data && typeof item.episode_data.r === 'number'
                );

                allEpisodes.push(...validBatch);
            }

            // 按奖励值排序并获取前N个
            allEpisodes.sort((a, b) => b.episode_data.r - a.episode_data.r);
            return allEpisodes.slice(0, topN);

        } catch (error) {
            return [];
        }
    }

    /**
     * 获取训练统计信息
     * @param {string} trainingId - 训练会话ID
     * @returns {Promise<Object>} 训练统计信息
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

            // 获取关键Episode数据进行统计
            const keyEpisodes = [];
            const totalEpisodes = trainingInfo.detail_length;
            const sampleSize = Math.min(100, totalEpisodes); // 采样100个Episode进行统计

            // 均匀采样
            const step = Math.floor(totalEpisodes / sampleSize);
            const sampleIndices = [];
            for (let i = 0; i < totalEpisodes; i += step) {
                sampleIndices.push(i);
            }

            // 确保包含最佳Episode和最后一个Episode
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

            // 计算统计信息
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
     * 检查历史服务健康状态
     * @returns {Promise<Object>} 健康状态响应
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
     * 验证训练ID是否存在
     * @param {string} trainingId - 训练会话ID
     * @returns {Promise<boolean>} 是否存在
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
     * 获取训练会话的元数据摘要
     * @param {Array<string>} trainingIds - 训练ID列表
     * @returns {Promise<Array>} 元数据摘要数组
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
     * 工具方法：计算中位数
     * @param {Array<number>} values - 数值数组
     * @returns {number} 中位数
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
     * 工具方法：格式化训练显示名称
     * @param {string} trainingId - 训练ID
     * @returns {Object} 格式化后的显示信息
     */
    formatTrainingDisplayName(trainingId) {
        const parts = trainingId.split('_');
        let algorithm = 'SAC';
        let timestamp = '';
        let mesh = '';

        try {
            if (parts.length >= 4) {
                if (parts[0] === 'continue') {
                    // Continue格式: continue_checkpointName_date_time_meshName
                    algorithm = 'Continue';
                    if (parts.length >= 5) {
                        // 找到日期时间部分（8位数字_6位数字的模式）
                        let dateTimeFound = false;
                        for (let i = 1; i < parts.length - 1; i++) {
                            if (parts[i].length === 8 && /^\d{8}$/.test(parts[i]) &&
                                i + 1 < parts.length && parts[i + 1].length === 6 && /^\d{6}$/.test(parts[i + 1])) {
                                // 找到了日期时间部分
                                timestamp = `${parts[i]}_${parts[i + 1]}`;
                                // mesh名称是剩余的部分
                                const beforeDateTime = parts.slice(1, i);
                                const afterDateTime = parts.slice(i + 2);
                                mesh = [...afterDateTime, ...beforeDateTime].join('_');
                                dateTimeFound = true;
                                break;
                            }
                        }

                        if (!dateTimeFound) {
                            // 回退到旧逻辑
                            timestamp = parts.length >= 4 ? `${parts[2]}_${parts[3]}` : '';
                            mesh = parts.slice(4).join('_');
                        }
                    } else {
                        // 不够5个部分，直接用剩余的作为mesh名
                        mesh = parts.slice(1).join('_');
                        timestamp = '';
                    }
                } else {
                    // 普通格式: algorithm_date_time_meshName
                    algorithm = parts[0].toUpperCase();
                    if (parts.length >= 3 && parts[1].length === 8 && /^\d{8}$/.test(parts[1]) &&
                        parts[2].length === 6 && /^\d{6}$/.test(parts[2])) {
                        timestamp = `${parts[1]}_${parts[2]}`;
                        mesh = parts.slice(3).join('_');
                    } else {
                        // 回退处理
                        timestamp = parts.slice(1, 3).join('_');
                        mesh = parts.slice(3).join('_');
                    }
                }
            } else {
                // 少于4个部分，无法正确解析
                algorithm = 'Unknown';
                timestamp = '';
                mesh = trainingId;
            }
        } catch (error) {
            // 解析出错，使用原始值
            console.warn('训练ID解析失败:', trainingId, error);
            algorithm = 'Unknown';
            timestamp = '';
            mesh = trainingId;
        }

        // 格式化时间戳显示
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
     * 导出Episode数据为CSV格式
     * @param {string} trainingId - 训练会话ID
     * @param {Array<number>} episodeIndices - Episode索引数组（可选，默认导出所有）
     * @returns {Promise<string>} CSV格式的数据
     */
    async exportEpisodesToCSV(trainingId, episodeIndices = null) {
        try {
            let episodeData;

            if (episodeIndices) {
                episodeData = await this.getBatchEpisodeData(trainingId, episodeIndices);
            } else {
                // 导出所有Episode
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

            // 生成CSV头部
            const headers = [
                'episode_index',
                'reward',
                'length',
                'completed',
                'boundary_vertices_count',
                'mesh_vertices_count'
            ];

            // 生成CSV内容
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
            throw new Error(`导出CSV失败: ${error.message}`);
        }
    }
}