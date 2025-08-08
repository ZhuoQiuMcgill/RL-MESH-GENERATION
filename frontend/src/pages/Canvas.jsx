import { Link } from 'react-router-dom'

const Canvas = () => {
  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-6">
        <Link 
          to="/"
          className="inline-flex items-center text-text-secondary hover:text-text-primary transition-colors mb-4"
        >
          ← Back to Dashboard
        </Link>
      </div>

      <div className="mb-8">
        <h2 className="text-3xl font-bold text-text-primary mb-4 flex items-center gap-3">
          <span className="text-4xl">🎨</span>
          3D Canvas
        </h2>
        <p className="text-text-secondary text-lg">
          Interactive 3D visualization and editing canvas for mesh inspection.
        </p>
      </div>

      <div className="bg-card border border-border-custom rounded-xl p-6">
        <h3 className="text-xl font-semibold text-text-primary mb-4">Coming Soon</h3>
        <p className="text-text-secondary">
          This page will feature an interactive 3D canvas for mesh visualization.
        </p>
      </div>
    </div>
  )
}

export default Canvas
