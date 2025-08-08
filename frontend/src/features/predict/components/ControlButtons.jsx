import React, { useState, useCallback } from 'react';
import Button from '../../../components/Button';
import { usePredictSession, PredictSessionStatus } from '../contexts/PredictSessionContext';
import predictApi from '../../../shared/api/predict';

/**
 * ControlButtons 组件
 * 提供预测会话的控制按钮：Next/Prev/Process All/Reset/Delete
 */
const ControlButtons = ({ className = '' }) => {
  const { 
    sessionId, 
    status, 
    currentStep, 
    totalSteps,
    actions 
  } = usePredictSession();

  // 各按钮的加载状态
  const [loadingStates, setLoadingStates] = useState({
    next: false,
    prev: false,
    processAll: false,
    reset: false,
    delete: false
  });

  // 设置单个按钮的加载状态
  const setButtonLoading = useCallback((buttonType, loading) => {
    setLoadingStates(prev => ({
      ...prev,
      [buttonType]: loading
    }));
  }, []);

  // 通用错误处理
  const handleError = useCallback((error, operation) => {
    console.error(`${operation} failed:`, error);
    actions.setError(error);
    actions.addLog({
      level: 'error',
      message: `${operation} 操作失败: ${error.message}`,
      data: { error: error.message }
    });
  }, [actions]);

  // Next 按钮处理函数
  const handleNext = useCallback(async () => {
    if (!sessionId) {
      actions.addLog({
        level: 'warning',
        message: 'No active session for next step'
      });
      return;
    }

    setButtonLoading('next', true);
    
    try {
      actions.addLog({
        level: 'info',
        message: `执行下一步 (步骤 ${currentStep + 1})`
      });

      const result = await predictApi.nextStep(sessionId, {
        currentStep,
        timestamp: new Date().toISOString()
      });

      // 更新会话状态
      actions.nextStep({
        meshData: result.meshData,
        stepData: result.stepData,
        quality: result.quality
      });

      actions.addLog({
        level: 'success',
        message: `步骤 ${currentStep + 1} 完成`,
        data: {
          step: currentStep + 1,
          quality: result.quality,
          executionTime: result.executionTime
        }
      });

    } catch (error) {
      handleError(error, 'Next Step');
    } finally {
      setButtonLoading('next', false);
    }
  }, [sessionId, currentStep, actions, handleError, setButtonLoading]);

  // Prev 按钮处理函数
  const handlePrev = useCallback(async () => {
    if (!sessionId) {
      actions.addLog({
        level: 'warning',
        message: 'No active session for previous step'
      });
      return;
    }

    setButtonLoading('prev', true);
    
    try {
      actions.addLog({
        level: 'info',
        message: `回退到上一步 (步骤 ${currentStep - 1})`
      });

      const result = await predictApi.prevStep(sessionId, {
        currentStep,
        timestamp: new Date().toISOString()
      });

      // 更新会话状态
      actions.nextStep({
        meshData: result.meshData,
        stepData: result.stepData,
        quality: result.quality
      });

      actions.addLog({
        level: 'success',
        message: `回退到步骤 ${currentStep - 1} 完成`,
        data: {
          step: currentStep - 1,
          quality: result.quality
        }
      });

    } catch (error) {
      handleError(error, 'Previous Step');
    } finally {
      setButtonLoading('prev', false);
    }
  }, [sessionId, currentStep, actions, handleError, setButtonLoading]);

  // Process All 按钮处理函数
  const handleProcessAll = useCallback(async () => {
    if (!sessionId) {
      actions.addLog({
        level: 'warning',
        message: 'No active session for process all'
      });
      return;
    }

    setButtonLoading('processAll', true);
    
    try {
      actions.addLog({
        level: 'info',
        message: '开始处理所有步骤...'
      });

      // 更新状态为运行中
      actions.startPrediction({
        mode: 'processAll',
        timestamp: new Date().toISOString()
      });

      const result = await predictApi.processAll(sessionId, {
        fromStep: currentStep,
        configuration: {
          logProgress: true,
          batchSize: 10
        }
      });

      // 如果是流式响应，循环处理结果
      if (result.stream) {
        let stepCount = currentStep;
        
        for await (const stepResult of result.stream) {
          stepCount++;
          
          // 更新进度
          actions.nextStep({
            meshData: stepResult.meshData,
            stepData: stepResult.stepData,
            quality: stepResult.quality
          });

          // 写入日志
          actions.addLog({
            level: 'info',
            message: `处理步骤 ${stepCount} 完成`,
            data: {
              step: stepCount,
              quality: stepResult.quality,
              progress: `${stepCount}/${totalSteps || stepCount}`,
              executionTime: stepResult.executionTime
            }
          });

          // 短暂延迟以避免UI阻塞
          await new Promise(resolve => setTimeout(resolve, 50));
        }
      } else {
        // 非流式响应，直接处理结果
        actions.updateProgress({
          currentStep: result.finalStep,
          totalSteps: result.totalSteps,
          progress: 100
        });

        // 批量写入日志
        result.steps?.forEach((stepData, index) => {
          actions.addLog({
            level: 'info',
            message: `处理步骤 ${currentStep + index + 1} 完成`,
            data: {
              step: currentStep + index + 1,
              quality: stepData.quality,
              executionTime: stepData.executionTime
            }
          });
        });
      }

      actions.addLog({
        level: 'success',
        message: `所有步骤处理完成，共处理 ${result.processedSteps} 个步骤`,
        data: {
          totalProcessed: result.processedSteps,
          finalQuality: result.finalQuality,
          totalTime: result.totalExecutionTime
        }
      });

    } catch (error) {
      handleError(error, 'Process All');
      // 处理失败时重置状态
      actions.pausePrediction();
    } finally {
      setButtonLoading('processAll', false);
    }
  }, [sessionId, currentStep, totalSteps, actions, handleError, setButtonLoading]);

  // Reset 按钮处理函数
  const handleReset = useCallback(async () => {
    if (!sessionId) {
      actions.resetSession();
      return;
    }

    setButtonLoading('reset', true);
    
    try {
      actions.addLog({
        level: 'info',
        message: '重置预测会话...'
      });

      await predictApi.resetSession(sessionId);

      actions.resetSession();
      
      actions.addLog({
        level: 'success',
        message: '预测会话已重置'
      });

    } catch (error) {
      handleError(error, 'Reset Session');
      // 即使API调用失败，也尝试重置本地状态
      actions.resetSession();
    } finally {
      setButtonLoading('reset', false);
    }
  }, [sessionId, actions, handleError, setButtonLoading]);

  // Delete 按钮处理函数
  const handleDelete = useCallback(async () => {
    if (!sessionId) {
      actions.addLog({
        level: 'warning',
        message: 'No active session to delete'
      });
      return;
    }

    // 确认删除
    const confirmed = window.confirm('确定要删除当前预测会话吗？此操作无法撤销。');
    if (!confirmed) {
      return;
    }

    setButtonLoading('delete', true);
    
    try {
      actions.addLog({
        level: 'info',
        message: '删除预测会话...'
      });

      await predictApi.deleteSession(sessionId);

      actions.addLog({
        level: 'success',
        message: '预测会话已删除'
      });

      // 删除成功后重置本地状态
      actions.resetSession();

    } catch (error) {
      handleError(error, 'Delete Session');
    } finally {
      setButtonLoading('delete', false);
    }
  }, [sessionId, actions, handleError, setButtonLoading]);

  // 按钮禁用逻辑
  const isButtonDisabled = useCallback((buttonType) => {
    // 如果对应按钮正在加载，则禁用
    if (loadingStates[buttonType]) {
      return true;
    }

    // 根据会话状态和按钮类型确定是否禁用
    switch (buttonType) {
      case 'next':
        return !sessionId || 
               status === PredictSessionStatus.RUNNING ||
               status === PredictSessionStatus.INITIALIZING ||
               (status === PredictSessionStatus.COMPLETED && currentStep >= totalSteps);
      
      case 'prev':
        return !sessionId || 
               status === PredictSessionStatus.RUNNING ||
               status === PredictSessionStatus.INITIALIZING ||
               currentStep <= 0;
      
      case 'processAll':
        return !sessionId || 
               status === PredictSessionStatus.RUNNING ||
               status === PredictSessionStatus.INITIALIZING ||
               status === PredictSessionStatus.COMPLETED;
      
      case 'reset':
        return loadingStates.delete || loadingStates.processAll; // 删除或处理中时禁用重置
      
      case 'delete':
        return status === PredictSessionStatus.RUNNING ||
               status === PredictSessionStatus.INITIALIZING ||
               loadingStates.processAll; // 运行中或处理中时禁用删除
      
      default:
        return false;
    }
  }, [sessionId, status, currentStep, totalSteps, loadingStates]);

  // 渲染加载图标
  const renderLoadingIcon = () => (
    <svg className="animate-spin -ml-1 mr-2 h-4 w-4" fill="none" viewBox="0 0 24 24">
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      />
    </svg>
  );

  return (
    <div className={`flex flex-wrap gap-2 ${className}`}>
      {/* Prev 按钮 */}
      <Button
        variant="outline"
        size="sm"
        disabled={isButtonDisabled('prev')}
        onClick={handlePrev}
        className="flex-shrink-0"
      >
        {loadingStates.prev && renderLoadingIcon()}
        <span className="mr-1">←</span>
        Prev
      </Button>

      {/* Next 按钮 */}
      <Button
        variant="primary"
        size="sm"
        disabled={isButtonDisabled('next')}
        onClick={handleNext}
        className="flex-shrink-0"
      >
        {loadingStates.next && renderLoadingIcon()}
        Next
        <span className="ml-1">→</span>
      </Button>

      {/* Process All 按钮 */}
      <Button
        variant="secondary"
        size="sm"
        disabled={isButtonDisabled('processAll')}
        onClick={handleProcessAll}
        className="flex-shrink-0"
      >
        {loadingStates.processAll && renderLoadingIcon()}
        Process All
      </Button>

      {/* Reset 按钮 */}
      <Button
        variant="outline"
        size="sm"
        disabled={isButtonDisabled('reset')}
        onClick={handleReset}
        className="flex-shrink-0"
      >
        {loadingStates.reset && renderLoadingIcon()}
        Reset
      </Button>

      {/* Delete 按钮 */}
      <Button
        variant="danger"
        size="sm"
        disabled={isButtonDisabled('delete')}
        onClick={handleDelete}
        className="flex-shrink-0"
      >
        {loadingStates.delete && renderLoadingIcon()}
        Delete
      </Button>
    </div>
  );
};

export default ControlButtons;
