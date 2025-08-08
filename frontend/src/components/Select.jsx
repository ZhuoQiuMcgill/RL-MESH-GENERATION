import React from 'react';

const Select = ({
  value,
  onChange,
  options = [],
  placeholder = "请选择...",
  disabled = false,
  className = '',
  loading = false,
  ...props
}) => {
  const baseClasses = 'w-full min-h-[44px] px-4 py-2 border border-gray-300 rounded-md bg-white text-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors duration-200';
  
  const disabledClasses = 'bg-gray-100 text-gray-400 cursor-not-allowed';
  
  const selectClasses = [
    baseClasses,
    disabled && disabledClasses,
    className
  ].filter(Boolean).join(' ');

  const handleChange = (e) => {
    if (!disabled && onChange) {
      onChange(e.target.value);
    }
  };

  return (
    <div className="relative">
      <select
        value={value}
        onChange={handleChange}
        disabled={disabled || loading}
        className={selectClasses}
        {...props}
      >
        <option value="">
          {loading ? "加载中..." : placeholder}
        </option>
        {options.map((option, index) => (
          <option
            key={option.value || index}
            value={option.value}
            title={option.description}
          >
            {option.label}
          </option>
        ))}
      </select>
      {loading && (
        <div className="absolute inset-y-0 right-3 flex items-center pointer-events-none">
          <div className="w-4 h-4 border-2 border-gray-300 border-t-blue-500 rounded-full animate-spin"></div>
        </div>
      )}
    </div>
  );
};

export default Select;
