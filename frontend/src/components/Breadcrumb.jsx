import { Link, useLocation } from 'react-router-dom'
import { getBreadcrumbLabel } from '../app/routes'

const Breadcrumb = () => {
  const location = useLocation()
  const pathSegments = location.pathname.split('/').filter(Boolean)

  return (
    <nav className="flex items-center gap-2 text-sm text-text-secondary mb-6">
      <Link 
        to="/" 
        className="hover:text-text-primary transition-colors"
      >
        🏠 Dashboard
      </Link>
      
      {pathSegments.length > 0 && (
        <>
          <span>→</span>
          {pathSegments.map((segment, index) => {
            const path = '/' + pathSegments.slice(0, index + 1).join('/')
            const isLast = index === pathSegments.length - 1
            const label = getBreadcrumbLabel(path) || segment
            
            return isLast ? (
              <span key={path} className="text-text-primary font-medium">
                {label}
              </span>
            ) : (
              <Link 
                key={path} 
                to={path}
                className="hover:text-text-primary transition-colors"
              >
                {label}
              </Link>
            )
          })}
        </>
      )}
    </nav>
  )
}

export default Breadcrumb
