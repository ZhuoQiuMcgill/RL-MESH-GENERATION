import React from 'react';
import { Link } from 'react-router-dom';
import { useTheme } from '../contexts/ThemeContext';
import '../styles/dashboard.css';

const Dashboard = () => {
  const { toggleTheme, isDark } = useTheme();

  const featureModules = [
    {
      title: "Predict",
      description: "Generate mesh predictions using trained RL models",
      path: "/predict",
      status: "ready",
      icon: (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
          <path d="M9 11H5a2 2 0 0 0-2 2v3c0 1.1.9 2 2 2h4" />
          <path d="M19 11h-4a2 2 0 0 0-2 2v3c0 1.1.9 2 2 2h4a2 2 0 0 0 2-2v-3a2 2 0 0 0-2-2z" />
          <path d="M11 5H7a2 2 0 0 0-2 2v2" />
          <path d="M17 5h4a2 2 0 0 1 2 2v2" />
          <circle cx="12" cy="12" r="3" />
        </svg>
      )
    },
    {
      title: "Train", 
      description: "Train reinforcement learning models for mesh generation",
      path: "/train",
      status: "ready",
      icon: (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
          <path d="M12 2L2 7l10 5 10-5-10-5z" />
          <path d="M2 17l10 5 10-5" />
          <path d="M2 12l10 5 10-5" />
        </svg>
      )
    },
    {
      title: "History",
      description: "View training history and analyze model performance",
      path: "/history", 
      status: "ready",
      icon: (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
          <path d="M3 3v5h5" />
          <path d="M3.05 13A9 9 0 1 0 6 5.3L3 8" />
          <path d="M12 7v5l4 2" />
        </svg>
      )
    }
  ];

  return (
    <div className="dashboard-container">
      {/* Feature Modules */}
      <div className="modules-section">
        <div className="modules-grid">
          {featureModules.map((module, index) => (
            <Link
              key={index}
              to={module.path}
              className="module-card"
            >
              <div className="module-icon">
                {module.icon}
              </div>
              <div className="module-content">
                <h3 className="module-title">{module.title}</h3>
                <p className="module-description">{module.description}</p>
              </div>
              <div className="module-status">
                <div className={`status-dot status-${module.status}`}></div>
                <span className="status-text">Ready</span>
              </div>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
