import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useTheme } from '../../contexts/ThemeContext';

const Header = () => {
  const location = useLocation();
  const { toggleTheme, isDark } = useTheme();
  const isDashboard = location.pathname === '/' || location.pathname === '/dashboard';

  const navItems = [
    { path: '/predict', label: 'Predict' },
    { path: '/train', label: 'Train' },
    { path: '/history', label: 'History' }
  ];

  const isActive = (path) => location.pathname === path;

  // Dashboard header with modern design
  if (isDashboard) {
    return (
      <header className="dashboard-header" style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '3rem',
        padding: '2rem 0 3rem 0',
        position: 'relative',
        background: 'var(--color-bg-primary)',
        borderBottom: '1px solid transparent',
        borderImage: 'linear-gradient(90deg, transparent 0%, var(--color-border-primary) 20%, var(--color-primary-start) 50%, var(--color-border-primary) 80%, transparent 100%) 1'
      }}>
        <div className="header-content" style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          gap: '0.5rem'
        }}>
          <h1 className="dashboard-title" style={{
            fontSize: '3.5rem',
            fontWeight: 200,
            margin: 0,
            background: 'linear-gradient(135deg, #030067 0%, #57007c 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
            lineHeight: 1.1,
            letterSpacing: '-0.02em',
            position: 'relative'
          }}>
            RL Mesh Generation
            <span className="header-accent" style={{
              display: 'inline-block',
              marginLeft: '0.5rem',
              padding: '0.25rem 0.75rem',
              background: 'linear-gradient(135deg, rgba(3, 0, 103, 0.1) 0%, rgba(87, 0, 124, 0.1) 100%)',
              borderRadius: '20px',
              fontSize: '0.75rem',
              fontWeight: 500,
              color: '#030067',
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
              border: '1px solid rgba(3, 0, 103, 0.2)'
            }}>AI Platform</span>
          </h1>
          <p className="dashboard-subtitle" style={{
            fontSize: '1.25rem',
            color: 'var(--color-text-secondary)',
            margin: 0,
            fontWeight: 300,
            lineHeight: 1.5,
            opacity: 0.9
          }}>Advanced reinforcement learning for intelligent mesh generation and optimization</p>
        </div>
        <div className="theme-toggle" style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem'
        }}>
          <span className="theme-label" style={{
            fontSize: '0.875rem',
            color: 'var(--color-text-tertiary)',
            fontWeight: 500,
            opacity: 0.8
          }}>{isDark ? 'Dark' : 'Light'}</span>
          <button 
            className="theme-btn" 
            onClick={toggleTheme}
            title={`Switch to ${isDark ? 'light' : 'dark'} theme`}
            style={{
              background: 'var(--color-bg-card)',
              border: '1px solid var(--color-border-primary)',
              borderRadius: '24px',
              width: '56px',
              height: '56px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: 'pointer',
              transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
              boxShadow: 'var(--shadow-sm)',
              position: 'relative',
              overflow: 'hidden'
            }}
            onMouseEnter={(e) => {
              e.target.style.borderColor = 'var(--color-primary-start)';
              e.target.style.transform = 'translateY(-3px) scale(1.05)';
              e.target.style.boxShadow = 'var(--shadow-xl)';
            }}
            onMouseLeave={(e) => {
              e.target.style.borderColor = 'var(--color-border-primary)';
              e.target.style.transform = 'translateY(0) scale(1)';
              e.target.style.boxShadow = 'var(--shadow-sm)';
            }}
          >
            <div className="theme-icon" style={{
              color: 'var(--color-text-primary)',
              transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)'
            }}>
              {isDark ? (
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="5"/>
                  <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/>
                </svg>
              ) : (
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
                </svg>
              )}
            </div>
          </button>
        </div>
      </header>
    );
  }

  // Regular header for other pages
  return (
    <header className="bg-white shadow-sm border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo/Brand */}
          <div className="flex-shrink-0">
            <Link to="/" className="text-xl font-bold text-gray-900 hover:text-blue-600 transition-colors">
              RL Mesh Generation
            </Link>
          </div>

          {/* Navigation Links */}
          <nav className="hidden md:flex space-x-8">
            {navItems.map(({ path, label }) => (
              <Link
                key={path}
                to={path}
                className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  isActive(path)
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                }`}
              >
                {label}
              </Link>
            ))}
          </nav>

          {/* Theme Toggle for non-dashboard pages */}
          <div className="flex items-center space-x-4">
            <button 
              onClick={toggleTheme}
              className="p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100 transition-colors"
              title={`Switch to ${isDark ? 'light' : 'dark'} theme`}
            >
              {isDark ? (
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="5"/>
                  <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/>
                </svg>
              ) : (
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
                </svg>
              )}
            </button>
            
            {/* Mobile menu button */}
            <div className="md:hidden">
              <button
                type="button"
                className="bg-white p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-blue-500"
                aria-expanded="false"
              >
                <span className="sr-only">Open main menu</span>
                <svg
                  className="h-6 w-6"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  aria-hidden="true"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 6h16M4 12h16M4 18h16"
                  />
                </svg>
              </button>
            </div>
          </div>
        </div>

        {/* Mobile menu - You can expand this with state management if needed */}
        <div className="md:hidden">
          <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3">
            {navItems.map(({ path, label }) => (
              <Link
                key={path}
                to={path}
                className={`block px-3 py-2 rounded-md text-base font-medium transition-colors ${
                  isActive(path)
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-50'
                }`}
              >
                {label}
              </Link>
            ))}
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
