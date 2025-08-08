import React, { useMemo } from 'react';
import { usePredictSession } from '../../contexts/PredictSessionContext';

const QualityPanel = () => {
  const { 
    meshData,
    logs,
    status
  } = usePredictSession();

  // Calculate average quality from mesh data and logs
  const qualityMetrics = useMemo(() => {
    let qualities = [];
    let totalElements = 0;
    let minQuality = Infinity;
    let maxQuality = -Infinity;

    // Extract quality data from logs
    const qualityLogs = logs.filter(log => 
      log.data && (
        log.data.quality !== undefined || 
        log.data.quality_score !== undefined ||
        log.data.element_quality !== undefined ||
        (log.data.meshData && log.data.meshData.quality)
      )
    );

    qualityLogs.forEach(log => {
      const data = log.data;
      let qualityValue = null;

      if (data.quality !== undefined) {
        qualityValue = data.quality;
      } else if (data.quality_score !== undefined) {
        qualityValue = data.quality_score;
      } else if (data.element_quality !== undefined) {
        qualityValue = data.element_quality;
      } else if (data.meshData && data.meshData.quality !== undefined) {
        qualityValue = data.meshData.quality;
      }

      if (qualityValue !== null && !isNaN(qualityValue)) {
        qualities.push(qualityValue);
        minQuality = Math.min(minQuality, qualityValue);
        maxQuality = Math.max(maxQuality, qualityValue);
      }
    });

    // Extract quality from current mesh data
    if (meshData) {
      if (meshData.elements && Array.isArray(meshData.elements)) {
        totalElements = meshData.elements.length;
        
        // If elements have quality scores
        meshData.elements.forEach(element => {
          if (element.quality !== undefined && !isNaN(element.quality)) {
            qualities.push(element.quality);
            minQuality = Math.min(minQuality, element.quality);
            maxQuality = Math.max(maxQuality, element.quality);
          }
        });
      }

      // Overall mesh quality
      if (meshData.quality !== undefined && !isNaN(meshData.quality)) {
        qualities.push(meshData.quality);
        minQuality = Math.min(minQuality, meshData.quality);
        maxQuality = Math.max(maxQuality, meshData.quality);
      }

      // Quality metrics
      if (meshData.qualityMetrics) {
        const metrics = meshData.qualityMetrics;
        ['average', 'min', 'max', 'median'].forEach(key => {
          if (metrics[key] !== undefined && !isNaN(metrics[key])) {
            qualities.push(metrics[key]);
            minQuality = Math.min(minQuality, metrics[key]);
            maxQuality = Math.max(maxQuality, metrics[key]);
          }
        });
      }
    }

    // Calculate statistics
    const averageQuality = qualities.length > 0 
      ? qualities.reduce((sum, q) => sum + q, 0) / qualities.length 
      : 0;

    const sortedQualities = [...qualities].sort((a, b) => a - b);
    const medianQuality = sortedQualities.length > 0
      ? sortedQualities.length % 2 === 0
        ? (sortedQualities[sortedQualities.length / 2 - 1] + sortedQualities[sortedQualities.length / 2]) / 2
        : sortedQualities[Math.floor(sortedQualities.length / 2)]
      : 0;

    return {
      average: averageQuality,
      min: minQuality === Infinity ? 0 : minQuality,
      max: maxQuality === -Infinity ? 0 : maxQuality,
      median: medianQuality,
      count: qualities.length,
      totalElements,
      distribution: qualities
    };
  }, [meshData, logs]);

  const getQualityColor = (quality) => {
    if (quality >= 0.8) return 'text-green-600';
    if (quality >= 0.6) return 'text-blue-600';
    if (quality >= 0.4) return 'text-yellow-600';
    if (quality >= 0.2) return 'text-orange-600';
    return 'text-red-600';
  };

  const getQualityBgColor = (quality) => {
    if (quality >= 0.8) return 'bg-green-500';
    if (quality >= 0.6) return 'bg-blue-500';
    if (quality >= 0.4) return 'bg-yellow-500';
    if (quality >= 0.2) return 'bg-orange-500';
    return 'bg-red-500';
  };

  const getQualityGrade = (quality) => {
    if (quality >= 0.9) return 'A+';
    if (quality >= 0.8) return 'A';
    if (quality >= 0.7) return 'B+';
    if (quality >= 0.6) return 'B';
    if (quality >= 0.5) return 'C+';
    if (quality >= 0.4) return 'C';
    if (quality >= 0.3) return 'D+';
    if (quality >= 0.2) return 'D';
    return 'F';
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-700">Quality Metrics</h3>
        <div className={`px-2 py-1 rounded text-xs font-medium ${
          qualityMetrics.average >= 0.8 ? 'bg-green-100 text-green-700' :
          qualityMetrics.average >= 0.6 ? 'bg-blue-100 text-blue-700' :
          qualityMetrics.average >= 0.4 ? 'bg-yellow-100 text-yellow-700' :
          'bg-red-100 text-red-700'
        }`}>
          {getQualityGrade(qualityMetrics.average)}
        </div>
      </div>
      
      <div className="space-y-3">
        {/* Average Quality */}
        <div>
          <div className="flex justify-between items-center mb-2">
            <span className="text-xs text-gray-600">Average Quality</span>
            <span className={`text-sm font-bold ${getQualityColor(qualityMetrics.average)}`}>
              {qualityMetrics.average.toFixed(3)}
            </span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div 
              className={`h-2 rounded-full transition-all duration-300 ${getQualityBgColor(qualityMetrics.average)}`}
              style={{ width: `${Math.min(qualityMetrics.average * 100, 100)}%` }}
            />
          </div>
        </div>

        {/* Quality Range */}
        <div className="grid grid-cols-2 gap-3 py-2 border-t border-gray-100">
          <div>
            <div className="flex justify-between items-center">
              <span className="text-xs text-gray-600">Min</span>
              <span className={`text-xs font-mono ${getQualityColor(qualityMetrics.min)}`}>
                {qualityMetrics.min.toFixed(3)}
              </span>
            </div>
          </div>
          <div>
            <div className="flex justify-between items-center">
              <span className="text-xs text-gray-600">Max</span>
              <span className={`text-xs font-mono ${getQualityColor(qualityMetrics.max)}`}>
                {qualityMetrics.max.toFixed(3)}
              </span>
            </div>
          </div>
        </div>

        {/* Median Quality */}
        <div className="flex justify-between items-center py-2 border-t border-gray-100">
          <span className="text-xs text-gray-600">Median</span>
          <span className={`text-xs font-mono ${getQualityColor(qualityMetrics.median)}`}>
            {qualityMetrics.median.toFixed(3)}
          </span>
        </div>

        {/* Element Count */}
        <div className="flex justify-between items-center py-2 border-t border-gray-100">
          <span className="text-xs text-gray-600">Elements</span>
          <div className="text-right">
            <div className="text-xs font-mono text-gray-800">{qualityMetrics.totalElements || 0}</div>
            <div className="text-xs text-gray-500">
              {qualityMetrics.count} with quality data
            </div>
          </div>
        </div>

        {/* Quality Distribution */}
        {qualityMetrics.count > 0 && (
          <div className="py-2 border-t border-gray-100">
            <div className="text-xs text-gray-600 mb-2">Quality Distribution</div>
            <div className="space-y-1">
              {[
                { label: 'Excellent (≥0.8)', min: 0.8, color: 'bg-green-500' },
                { label: 'Good (≥0.6)', min: 0.6, max: 0.8, color: 'bg-blue-500' },
                { label: 'Fair (≥0.4)', min: 0.4, max: 0.6, color: 'bg-yellow-500' },
                { label: 'Poor (<0.4)', max: 0.4, color: 'bg-red-500' }
              ].map((range, idx) => {
                const count = qualityMetrics.distribution.filter(q => {
                  if (range.min !== undefined && range.max !== undefined) {
                    return q >= range.min && q < range.max;
                  } else if (range.min !== undefined) {
                    return q >= range.min;
                  } else {
                    return q < range.max;
                  }
                }).length;
                
                const percentage = qualityMetrics.count > 0 ? (count / qualityMetrics.count) * 100 : 0;
                
                return (
                  <div key={idx} className="flex items-center space-x-2">
                    <div className={`w-3 h-3 rounded-full ${range.color}`} />
                    <div className="flex-1 flex justify-between items-center">
                      <span className="text-xs text-gray-600">{range.label}</span>
                      <span className="text-xs font-mono text-gray-800">
                        {count} ({percentage.toFixed(1)}%)
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Quality Trend */}
        {qualityMetrics.distribution.length > 1 && (
          <div className="py-2 border-t border-gray-100">
            <div className="text-xs text-gray-600 mb-2">Recent Trend</div>
            <div className="h-8 bg-gray-100 rounded overflow-hidden flex">
              {qualityMetrics.distribution.slice(-10).map((quality, idx) => (
                <div
                  key={idx}
                  className={`flex-1 ${getQualityBgColor(quality)} opacity-70`}
                  style={{ height: `${quality * 100}%`, alignSelf: 'flex-end' }}
                />
              ))}
            </div>
          </div>
        )}

        {/* Status */}
        <div className="pt-2 border-t border-gray-100">
          <div className="flex justify-between items-center">
            <span className="text-xs text-gray-600">Computation Status</span>
            <span className={`text-xs px-2 py-1 rounded ${
              qualityMetrics.count > 0 ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-700'
            }`}>
              {qualityMetrics.count > 0 ? 'Active' : 'No Data'}
            </span>
          </div>
        </div>

        {/* No Data State */}
        {qualityMetrics.count === 0 && (
          <div className="text-center py-4">
            <div className="text-xs text-gray-400 italic">
              No quality data available
            </div>
            <div className="text-xs text-gray-500 mt-1">
              Start a prediction to see quality metrics
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default QualityPanel;
