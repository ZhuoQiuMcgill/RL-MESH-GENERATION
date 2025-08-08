import React from 'react';
import SessionStatusPanel from './SessionStatusPanel';
import ActionInfoPanel from './ActionInfoPanel';
import ReferencePointPanel from './ReferencePointPanel';
import QualityPanel from './QualityPanel';

const StatusDisplay = ({ layout = 'grid', className = '' }) => {
  const layoutClasses = {
    grid: 'grid grid-cols-1 md:grid-cols-2 gap-4',
    column: 'space-y-4',
    row: 'flex flex-wrap gap-4'
  };

  const containerClass = `${layoutClasses[layout]} ${className}`;

  return (
    <div className={containerClass}>
      {/* Session Status Panel */}
      <div className="w-full">
        <SessionStatusPanel />
      </div>

      {/* Action Information Panel */}
      <div className="w-full">
        <ActionInfoPanel />
      </div>

      {/* Reference Point Panel */}
      <div className="w-full">
        <ReferencePointPanel />
      </div>

      {/* Quality Panel */}
      <div className="w-full">
        <QualityPanel />
      </div>
    </div>
  );
};

export default StatusDisplay;
