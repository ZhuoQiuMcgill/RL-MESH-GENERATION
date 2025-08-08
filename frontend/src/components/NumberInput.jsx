import React from 'react';

const NumberInput = ({
  value,
  onChange,
  min,
  max,
  step = 1,
  placeholder,
  disabled = false,
  className = '',
  label,
  ...props
}) => {
  const baseClasses = 'w-full px-3 py-2 border border-gray-300 rounded-md bg-white text-gray-700 text-center focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors duration-200 font-medium';
  
  const disabledClasses = 'bg-gray-100 text-gray-400 cursor-not-allowed';
  
  const inputClasses = [
    baseClasses,
    disabled && disabledClasses,
    className
  ].filter(Boolean).join(' ');

  const handleChange = (e) => {
    if (!disabled && onChange) {
      const newValue = e.target.value;
      // 允许空值用于编辑
      if (newValue === '') {
        onChange('');
        return;
      }
      
      const numValue = parseFloat(newValue);
      if (!isNaN(numValue)) {
        onChange(numValue);
      }
    }
  };

  const handleBlur = (e) => {
    if (!disabled && onChange) {
      const numValue = parseFloat(e.target.value);
      if (isNaN(numValue) || e.target.value === '') {
        // 如果输入无效，重置为最小值或默认值
        onChange(min !== undefined ? min : 0);
      }
    }
  };

  return (
    <div className="flex flex-col">
      {label && (
        <label className="text-xs font-semibold text-gray-700 mb-1">
          {label}
        </label>
      )}
      <input
        type="number"
        value={value}
        onChange={handleChange}
        onBlur={handleBlur}
        min={min}
        max={max}
        step={step}
        placeholder={placeholder}
        disabled={disabled}
        className={inputClasses}
        {...props}
      />
    </div>
  );
};

export default NumberInput;
