import React, { useState, useEffect, useReducer } from 'react';
import { Button, Select, NumberInput } from '../../../components';
import { listComponents } from '../../../shared/api/predict';
import { usePredictSession } from '../contexts/PredictSessionContext';

// Configuration reducer for complex state management
const configReducer = (state, action) => {
  switch (action.type) {
    case 'SET_COMPONENTS':
      return {
        ...state,
        components: action.payload,
        loading: false,
      };
    case 'SET_LOADING':
      return {
        ...state,
        loading: action.payload,
      };
    case 'SET_ERROR':
      return {
        ...state,
        error: action.payload,
        loading: false,
      };
    case 'UPDATE_CONFIG':
      return {
        ...state,
        config: {
          ...state.config,
          ...action.payload,
        },
      };
    case 'RESET_CONFIG':
      return {
        ...state,
        config: {
          mesh: '',
          predictor: '',
          refSelector: '',
          qualityMethod: '',
          predictorConfig: {
            modelPath: '',
            n: 2,
            g: 3,
            beta: 6,
          },
          refSelectorConfig: {
            n: 2,
          },
        },
      };
    default:
      return state;
  }
};

const initialState = {
  loading: true,
  error: null,
  components: null,
  config: {
    mesh: '',
    predictor: '',
    refSelector: '',
    qualityMethod: '',
    predictorConfig: {
      modelPath: '',
      n: 2,
      g: 3,
      beta: 6,
    },
    refSelectorConfig: {
      n: 2,
    },
  },
};

const ConfigurationPanel = ({ 
  disabled = false,
  className = '' 
}) => {
  const [state, localDispatch] = useReducer(configReducer, initialState);
  const { loading, error, components, config } = state;
  
  // Get session context for dispatching updates
  const { actions } = usePredictSession();

  // Load components on mount
  useEffect(() => {
    const loadData = async () => {
      try {
        localDispatch({ type: 'SET_LOADING', payload: true });
        const data = await listComponents();
        localDispatch({ type: 'SET_COMPONENTS', payload: data });
      } catch (err) {
        localDispatch({ type: 'SET_ERROR', payload: err.message || '加载组件失败' });
      }
    };

    loadData();
  }, []);

  // Check if configuration is valid for session creation
  const isConfigValid = () => {
    return config.mesh && 
           config.predictor && 
           config.refSelector && 
           config.qualityMethod &&
           (config.predictor !== 'RL' || config.predictorConfig.modelPath);
  };

  // Dispatch config changes to session context
  useEffect(() => {
    if (actions && actions.configUpdate) {
      actions.configUpdate({ ...config, isValid: isConfigValid() });
    }
  }, [config, actions]);

  // Handle form field changes
  const handleConfigChange = (field, value) => {
    localDispatch({
      type: 'UPDATE_CONFIG',
      payload: { [field]: value }
    });
  };

  const handlePredictorConfigChange = (field, value) => {
    localDispatch({
      type: 'UPDATE_CONFIG',
      payload: {
        predictorConfig: {
          ...config.predictorConfig,
          [field]: value
        }
      }
    });
  };

  const handleRefSelectorConfigChange = (field, value) => {
    localDispatch({
      type: 'UPDATE_CONFIG',
      payload: {
        refSelectorConfig: {
          ...config.refSelectorConfig,
          [field]: value
        }
      }
    });
  };

  // Prepare options for dropdowns
  const getMeshOptions = () => {
    if (!components?.initial_meshes) return [];
    return components.initial_meshes.map(mesh => ({
      value: mesh,
      label: mesh
    }));
  };

  const getPredictorOptions = () => {
    if (!components?.predictors) return [];
    return Object.entries(components.predictors).map(([key, predictor]) => ({
      value: key,
      label: `${predictor.name}`,
      description: predictor.description
    }));
  };

  const getRefSelectorOptions = () => {
    if (!components?.reference_selectors) return [];
    return Object.entries(components.reference_selectors).map(([key, selector]) => ({
      value: key,
      label: selector.name,
      description: selector.description
    }));
  };

  const getQualityMethodOptions = () => {
    if (!components?.quality_methods) return [];
    return components.quality_methods.map(method => ({
      value: method,
      label: method
    }));
  };

  const getModelOptions = () => {
    if (!components?.trained_models) return [];
    return components.trained_models.map(model => ({
      value: model.path,
      label: `${model.name} (${formatFileSize(model.size)})`
    }));
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    // The config update is already handled by useEffect
    // We can trigger additional actions here if needed
    if (isConfigValid() && actions.configUpdate) {
      actions.configUpdate({ ...config, isValid: true, submitted: true });
    }
  };

  const handleReset = () => {
    localDispatch({ type: 'RESET_CONFIG' });
  };

  if (error) {
    return (
      <div className={`p-6 ${className}`}>
        <div className="bg-red-50 border border-red-200 rounded-md p-4">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-red-800">
                加载错误
              </h3>
              <p className="mt-2 text-sm text-red-700">
                {error}
              </p>
              <div className="mt-3">
                <Button 
                  variant="outline"
                  size="sm"
                  onClick={() => window.location.reload()}
                >
                  刷新页面
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-white ${className}`}>
      {/* Header */}
      <div className="p-6 border-b border-gray-200">
        <h2 className="text-2xl font-bold text-gray-800">配置面板</h2>
        <p className="text-sm text-gray-600 mt-1">设置网格生成参数</p>
      </div>

      {/* Configuration Form */}
      <form onSubmit={handleSubmit} className="flex-1 p-6 space-y-6 overflow-y-auto">
        {/* Session Setup Section */}
        <div>
          <h3 className="text-lg font-semibold text-gray-800 mb-4">会话设置</h3>
          
          {/* Mesh Selection */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              选择初始网格
            </label>
            <Select
              value={config.mesh}
              onChange={(value) => handleConfigChange('mesh', value)}
              options={getMeshOptions()}
              placeholder="选择网格..."
              disabled={disabled}
              loading={loading}
            />
            {config.mesh && (
              <div className="mt-2 text-xs text-gray-500">
                已选择: {config.mesh}
              </div>
            )}
          </div>

          {/* Predictor Selection */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              选择预测器
            </label>
            <Select
              value={config.predictor}
              onChange={(value) => handleConfigChange('predictor', value)}
              options={getPredictorOptions()}
              placeholder="选择预测器..."
              disabled={disabled}
              loading={loading}
            />
            
            {/* Predictor Configuration */}
            {config.predictor && (
              <div className="mt-3 p-4 bg-gray-50 border border-gray-200 rounded-md">
                <h4 className="text-sm font-medium text-gray-700 mb-3">预测器配置</h4>
                
                {config.predictor === 'RL' && (
                  <div className="space-y-3">
                    {/* Model Path */}
                    <div>
                      <label className="block text-xs font-medium text-gray-700 mb-1">
                        模型路径
                      </label>
                      <Select
                        value={config.predictorConfig.modelPath}
                        onChange={(value) => handlePredictorConfigChange('modelPath', value)}
                        options={getModelOptions()}
                        placeholder="选择训练的模型..."
                        disabled={disabled}
                      />
                    </div>
                    
                    {/* Parameters Grid */}
                    <div className="grid grid-cols-3 gap-2">
                      <NumberInput
                        label="N"
                        value={config.predictorConfig.n}
                        onChange={(value) => handlePredictorConfigChange('n', value)}
                        min={1}
                        disabled={disabled}
                      />
                      <NumberInput
                        label="G"
                        value={config.predictorConfig.g}
                        onChange={(value) => handlePredictorConfigChange('g', value)}
                        min={1}
                        disabled={disabled}
                      />
                      <NumberInput
                        label="Beta"
                        value={config.predictorConfig.beta}
                        onChange={(value) => handlePredictorConfigChange('beta', value)}
                        min={1}
                        disabled={disabled}
                      />
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Reference Selector */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              参考选择器
            </label>
            <Select
              value={config.refSelector}
              onChange={(value) => handleConfigChange('refSelector', value)}
              options={getRefSelectorOptions()}
              placeholder="选择参考选择器..."
              disabled={disabled}
              loading={loading}
            />
            
            {/* Reference Selector Configuration */}
            {config.refSelector && config.refSelector !== 'default' && (
              <div className="mt-3 p-4 bg-gray-50 border border-gray-200 rounded-md">
                <NumberInput
                  label="N"
                  value={config.refSelectorConfig.n}
                  onChange={(value) => handleRefSelectorConfigChange('n', value)}
                  min={1}
                  disabled={disabled}
                />
              </div>
            )}
          </div>

          {/* Quality Method */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              质量评估方法
            </label>
            <Select
              value={config.qualityMethod}
              onChange={(value) => handleConfigChange('qualityMethod', value)}
              options={getQualityMethodOptions()}
              placeholder="选择质量方法..."
              disabled={disabled}
              loading={loading}
            />
          </div>

          {/* Action Buttons */}
          <div className="flex space-x-3">
            <Button
              type="submit"
              variant="primary"
              disabled={!isConfigValid() || disabled}
              className="flex-1"
            >
              创建会话
            </Button>
            <Button
              type="button"
              variant="secondary"
              onClick={handleReset}
              disabled={disabled}
            >
              重置
            </Button>
          </div>
        </div>
      </form>
    </div>
  );
};

export default ConfigurationPanel;
