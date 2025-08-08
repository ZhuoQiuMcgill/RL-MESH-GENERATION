import { lazy, Suspense } from 'react';
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';

import { ThemeProvider } from '../contexts/ThemeContext';
import Layout from '../components/Layout';

// Lazy-loaded page components
const Dashboard = lazy(() => import('../pages/Dashboard'));
const Predict = lazy(() => import('../pages/Predict'));
const Train = lazy(() => import('../pages/Train'));
const History = lazy(() => import('../pages/History'));

const AppRouter = () => {
  return (
    <ThemeProvider>
      <BrowserRouter>
        <Layout>
          <Suspense fallback={<div className="loading-spinner">Loading...</div>}>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/predict" element={<Predict />} />
              <Route path="/train" element={<Train />} />
              <Route path="/history" element={<History />} />
              <Route path="*" element={<div>404 - Page Not Found</div>} />
            </Routes>
          </Suspense>
        </Layout>
      </BrowserRouter>
    </ThemeProvider>
  );
};

export default AppRouter;
