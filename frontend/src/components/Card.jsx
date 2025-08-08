import React from 'react';

const Card = ({ 
  children, 
  title, 
  subtitle,
  variant = 'default',
  className = '',
  ...props 
}) => {
  const baseClasses = 'rounded-lg shadow-md overflow-hidden';
  
  const variants = {
    default: 'bg-white border border-gray-200',
    elevated: 'bg-white shadow-lg',
    outlined: 'bg-white border-2 border-gray-300',
    flat: 'bg-gray-50 shadow-none border border-gray-100'
  };
  
  const cardClasses = [
    baseClasses,
    variants[variant],
    className
  ].filter(Boolean).join(' ');

  return (
    <div className={cardClasses} {...props}>
      {(title || subtitle) && (
        <div className="px-6 py-4 border-b border-gray-200">
          {title && (
            <h3 className="text-lg font-semibold text-gray-900">
              {title}
            </h3>
          )}
          {subtitle && (
            <p className="text-sm text-gray-600 mt-1">
              {subtitle}
            </p>
          )}
        </div>
      )}
      <div className="p-6">
        {children}
      </div>
    </div>
  );
};

export default Card;
