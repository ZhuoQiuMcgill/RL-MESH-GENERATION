import React from 'react'
import { Link } from 'react-router-dom'
import { cn } from '../utils/cn'

const PageHeader = React.forwardRef(({ 
  className,
  title,
  subtitle,
  icon,
  actions,
  breadcrumbs,
  backLink,
  size = 'default', // 'sm', 'default', 'lg'
  ...props 
}, ref) => {
  const baseClasses = 'mb-8'
  
  const sizes = {
    sm: {
      title: 'text-2xl font-bold',
      subtitle: 'text-base',
      spacing: 'mb-6'
    },
    default: {
      title: 'text-3xl font-bold',
      subtitle: 'text-lg',
      spacing: 'mb-8'
    },
    lg: {
      title: 'text-4xl font-bold',
      subtitle: 'text-xl',
      spacing: 'mb-10'
    }
  }
  
  const sizeConfig = sizes[size] || sizes.default
  const classes = cn(baseClasses, sizeConfig.spacing, className)
  
  return (
    <div ref={ref} className={classes} {...props}>
      {/* Breadcrumbs or Back Link */}
      {(breadcrumbs || backLink) && (
        <div className="mb-6">
          {backLink && (
            <Link 
              to={backLink.href}
              className="inline-flex items-center text-text-secondary hover:text-text-primary transition-colors"
            >
              <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
              {backLink.label || 'Back'}
            </Link>
          )}
          
          {breadcrumbs && (
            <nav className="flex" aria-label="Breadcrumb">
              <ol className="inline-flex items-center space-x-1 md:space-x-3">
                {breadcrumbs.map((crumb, index) => (
                  <li key={index} className="inline-flex items-center">
                    {index > 0 && (
                      <svg className="w-4 h-4 text-text-secondary mx-1" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clipRule="evenodd" />
                      </svg>
                    )}
                    {crumb.href ? (
                      <Link
                        to={crumb.href}
                        className="inline-flex items-center text-sm font-medium text-text-secondary hover:text-text-primary transition-colors"
                      >
                        {crumb.icon && <span className="mr-2">{crumb.icon}</span>}
                        {crumb.label}
                      </Link>
                    ) : (
                      <span className="inline-flex items-center text-sm font-medium text-text-primary">
                        {crumb.icon && <span className="mr-2">{crumb.icon}</span>}
                        {crumb.label}
                      </span>
                    )}
                  </li>
                ))}
              </ol>
            </nav>
          )}
        </div>
      )}

      {/* Main Header Content */}
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <div className="flex items-center gap-3 mb-4">
            {icon && (
              <div className="text-4xl flex-shrink-0">
                {typeof icon === 'string' ? icon : icon}
              </div>
            )}
            <h1 className={cn(sizeConfig.title, 'text-text-primary')}>
              {title}
            </h1>
          </div>
          
          {subtitle && (
            <p className={cn(sizeConfig.subtitle, 'text-text-secondary max-w-3xl')}>
              {subtitle}
            </p>
          )}
        </div>

        {/* Actions */}
        {actions && (
          <div className="ml-6 flex-shrink-0">
            {Array.isArray(actions) ? (
              <div className="flex items-center gap-3">
                {actions.map((action, index) => (
                  <div key={index}>
                    {action}
                  </div>
                ))}
              </div>
            ) : (
              actions
            )}
          </div>
        )}
      </div>
    </div>
  )
})

PageHeader.displayName = 'PageHeader'

export default PageHeader
