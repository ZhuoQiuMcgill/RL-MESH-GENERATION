import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useTheme } from '../contexts/ThemeContext';
import '../App.css';

const Layout = ({ children }) => {
  const location = useLocation();
  const { toggleTheme, isDark } = useTheme();
  const isDashboard = location.pathname === '/' || location.pathname === '/dashboard';
  
  // Responsive state
  const [isMobile, setIsMobile] = useState(false);
  const [titleKey, setTitleKey] = useState(0);
  
  useEffect(() => {
    const checkScreenSize = () => {
      setIsMobile(window.innerWidth <= 768);
    };
    
    checkScreenSize();
    window.addEventListener('resize', checkScreenSize);
    return () => window.removeEventListener('resize', checkScreenSize);
  }, []);

  // Force re-render of title when theme changes
  useEffect(() => {
    setTitleKey(prev => prev + 1);
  }, [isDark]);

  // Get current page title for subtitle
  const getPageSubtitle = () => {
    switch (location.pathname) {
      case '/':
      case '/dashboard':
        return 'Advanced reinforcement learning for intelligent mesh generation and optimization';
      case '/predict':
        return 'Generate optimized mesh structures using trained reinforcement learning models';
      case '/train':
        return 'Train and fine-tune RL models with custom datasets and parameters';
      case '/history':
        return 'Review past experiments, analyze results, and track model performance';
      default:
        return 'Advanced reinforcement learning for intelligent mesh generation and optimization';
    }
  };

  // Navigation items - only dashboard
  const navItems = [
    { path: '/', label: 'Dashboard', active: isDashboard }
  ];

  // Modern layout for all pages
  return (
    <div className="app" style={{ 
      background: isDark ? '#0f1419' : '#ffffff',
      minHeight: '100vh',
      color: isDark ? '#f7fafc' : '#111827'
    }}>
      <header style={{
        width: '100%',
        background: isDark ? '#0f1419' : '#ffffff',
        position: 'relative',
        marginBottom: '2rem',
        borderBottom: '1px solid rgba(3, 0, 103, 0.1)'
      }}>
        {/* Gradient border line */}
        <div style={{
          position: 'absolute',
          bottom: 0,
          left: 0,
          right: 0,
          height: '1px',
          background: 'linear-gradient(90deg, transparent 0%, #e5e7eb 20%, #030067 50%, #e5e7eb 80%, transparent 100%)'
        }}></div>
        
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: isMobile ? '0.5rem 1rem' : '0.5rem 2rem',
          width: '100%',
          boxSizing: 'border-box',
          minHeight: '60px'
        }}>
        
        {/* Left side - Title and subtitle in a single line */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: isMobile ? '0.5rem' : '1rem',
          flex: 1,
          minWidth: 0
        }}>
          <h1 style={{
            fontSize: isMobile ? '1.2rem' : '1.6rem',
            fontWeight: 200,
            margin: 0,
            backgroundImage: isDark 
              ? 'linear-gradient(135deg, #60a5ff 0%, #d465ff 100%)'
              : 'linear-gradient(135deg, #030067 0%, #57007c 100%)',
            backgroundSize: '100%',
            backgroundRepeat: 'no-repeat',
            backgroundPosition: 'center',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
            MozBackgroundClip: 'text',
            MozTextFillColor: 'transparent',
            lineHeight: 1,
            letterSpacing: '-0.02em',
            whiteSpace: 'nowrap'
          }}>
            RL Mesh Generation
            <span style={{
              display: 'inline-block',
              marginLeft: isMobile ? '0.3rem' : '0.4rem',
              padding: '0.15rem 0.5rem',
              background: isDark
                ? 'linear-gradient(135deg, rgba(79, 134, 255, 0.15) 0%, rgba(180, 90, 255, 0.15) 100%)'
                : 'linear-gradient(135deg, rgba(3, 0, 103, 0.1) 0%, rgba(87, 0, 124, 0.1) 100%)',
              borderRadius: '12px',
              fontSize: isMobile ? '0.4rem' : '0.5rem',
              fontWeight: 500,
              color: isDark ? '#9bb3ff' : '#030067',
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
              border: isDark 
                ? '1px solid rgba(79, 134, 255, 0.3)'
                : '1px solid rgba(3, 0, 103, 0.2)'
            }}>AI Platform</span>
          </h1>
          
          {!isMobile && (
            <>
              <div style={{
                width: '1px',
                height: '30px',
                background: 'linear-gradient(180deg, transparent, rgba(3, 0, 103, 0.3), transparent)',
                margin: '0 0.5rem'
              }}></div>
              
              <p style={{
                fontSize: '0.8rem',
                color: isDark ? '#a0aec0' : '#6b7280',
                margin: 0,
                fontWeight: 300,
                lineHeight: 1.3,
                opacity: 0.8,
                maxWidth: '400px',
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis'
              }}>Advanced reinforcement learning for intelligent mesh generation</p>
            </>
          )}
          
          {/* Dashboard link for non-dashboard pages */}
          {!isDashboard && (
            <>
              <div style={{
                width: '1px',
                height: '20px',
                background: 'linear-gradient(180deg, transparent, rgba(3, 0, 103, 0.2), transparent)',
                margin: '0 0.75rem'
              }}></div>
              
              <Link
                to="/"
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  textDecoration: 'none',
                  padding: '0.3rem 0.8rem',
                  borderRadius: '20px',
                  fontSize: '0.75rem',
                  fontWeight: 500,
                  color: isDark ? '#a0aec0' : '#6b7280',
                  background: 'transparent',
                  border: '1px solid rgba(3, 0, 103, 0.15)',
                  transition: 'all 0.3s ease',
                  whiteSpace: 'nowrap'
                }}
                onMouseEnter={(e) => {
                  e.target.style.color = '#030067';
                  e.target.style.background = 'rgba(3, 0, 103, 0.08)';
                  e.target.style.borderColor = 'rgba(3, 0, 103, 0.3)';
                }}
                onMouseLeave={(e) => {
                  e.target.style.color = isDark ? '#a0aec0' : '#6b7280';
                  e.target.style.background = 'transparent';
                  e.target.style.borderColor = 'rgba(3, 0, 103, 0.15)';
                }}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="m15 18-6-6 6-6"/>
                </svg>
                Dashboard
              </Link>
            </>
          )}
        </div>
        
        {/* Mobile dashboard link */}
        {isMobile && !isDashboard && (
          <Link
            to="/"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.4rem',
              textDecoration: 'none',
              padding: '0.25rem 0.6rem',
              borderRadius: '16px',
              fontSize: '0.7rem',
              fontWeight: 500,
              color: isDark ? '#a0aec0' : '#6b7280',
              background: 'transparent',
              border: '1px solid rgba(3, 0, 103, 0.15)',
              transition: 'all 0.3s ease',
              whiteSpace: 'nowrap'
            }}
            onMouseEnter={(e) => {
              e.target.style.color = '#030067';
              e.target.style.background = 'rgba(3, 0, 103, 0.08)';
              e.target.style.borderColor = 'rgba(3, 0, 103, 0.3)';
            }}
            onMouseLeave={(e) => {
              e.target.style.color = isDark ? '#a0aec0' : '#6b7280';
              e.target.style.background = 'transparent';
              e.target.style.borderColor = 'rgba(3, 0, 103, 0.15)';
            }}
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="m15 18-6-6 6-6"/>
            </svg>
            Dashboard
          </Link>
        )}
        
        {/* Right side - Theme toggle */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem'
        }}>
          <span style={{
            fontSize: '0.875rem',
            color: isDark ? '#718096' : '#9ca3af',
            fontWeight: 500,
            opacity: 0.8
          }}>{isDark ? 'Dark' : 'Light'}</span>
          <button 
            onClick={toggleTheme}
            title={`Switch to ${isDark ? 'light' : 'dark'} theme`}
            style={{
              background: isDark ? '#343a4a' : '#ffffff',
              border: isDark ? '1px solid #4a5568' : '1px solid #e5e7eb',
              borderRadius: '24px',
              width: '56px',
              height: '56px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: 'pointer',
              transition: 'all 0.3s ease',
              boxShadow: isDark ? '0 1px 2px rgba(0, 0, 0, 0.3)' : '0 1px 2px 0 rgb(0 0 0 / 0.05)',
              position: 'relative',
              overflow: 'hidden',
              color: isDark ? '#f7fafc' : '#111827'
            }}
          >
            <div style={{
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
        </div>
      </header>
      <main style={{
        maxWidth: '1200px',
        margin: '0 auto',
        padding: '0 2rem 2rem 2rem',
        background: isDark ? '#0f1419' : '#ffffff',
        minHeight: 'calc(100vh - 200px)'
      }}>
        {children}
      </main>
    </div>
  );
};

export default Layout;
