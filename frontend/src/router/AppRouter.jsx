import { lazy, Suspense } from 'react';
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';

import { ThemeProvider } from '../contexts/ThemeContext';
import { PredictSessionProvider } from '../features/predict/contexts/PredictSessionContext';
import Layout from '../components/Layout';
import { LoadingOverlay, ErrorToast } from '../shared/components';
import { usePredictSession } from '../features/predict/contexts/PredictSessionContext';

// Lazy-loaded page components
const Dashboard = lazy(() => import('../pages/Dashboard'));
const Predict = lazy(() => import('../pages/Predict'));
const Train = lazy(() => import('../pages/Train'));
const History = lazy(() => import('../pages/History'));

// App内容组件
const AppContent = () => {
  const { error, loading, actions } = usePredictSession();

  return (
    <>
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
      
      {/* Global Components */}
      <LoadingOverlay isLoading={loading} />
      <ErrorToast 
        error={error} 
        onClose={actions.clearError}
      />
    </>
  );
};

const AppRouter = () => {
  return (
    <ThemeProvider>
      <BrowserRouter>
        <PredictSessionProvider>
          <AppContent />
        </PredictSessionProvider>
      </BrowserRouter>
    </ThemeProvider>
  );
};

export default AppRouter;
