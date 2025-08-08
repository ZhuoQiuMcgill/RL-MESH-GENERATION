import { Link, useLocation } from 'react-router-dom'
import { useTheme } from '../app/providers'
import { getNavigationItems } from '../app/routes'
import {
  SunIcon,
  MoonIcon,
  MonitorIcon
} from '../shared/icons'

const NavHeader = () => {
  const { isDark, toggleTheme, themePreference, THEME_OPTIONS } = useTheme()
  const location = useLocation()

  // Get theme button icon and tooltip based on preference
  const getThemeButton = () => {
    switch (themePreference) {
      case THEME_OPTIONS.LIGHT:
        return { IconComponent: SunIcon, tooltip: 'Light mode' }
      case THEME_OPTIONS.DARK:
        return { IconComponent: MoonIcon, tooltip: 'Dark mode' }
      case THEME_OPTIONS.SYSTEM:
        return { IconComponent: MonitorIcon, tooltip: `System (${isDark ? 'dark' : 'light'})` }
      default:
        return { IconComponent: isDark ? SunIcon : MoonIcon, tooltip: 'Toggle theme' }
    }
  }

  const { IconComponent, tooltip } = getThemeButton()

  const navItems = getNavigationItems()

  return (
    /* Header with gradient background */
    <header className="gradient-bg p-6 rounded-xl mb-8">
      <div className="flex justify-between items-center">
        <div className="text-center flex-1">
          <h1 className="text-4xl font-bold text-white mb-2">RL Mesh Generation</h1>
          <p className="text-white/80">Reinforcement Learning for 3D Mesh Generation</p>
        </div>
        <button 
          onClick={toggleTheme}
          title={tooltip}
          className="text-white/80 hover:text-white transition-colors p-2 rounded-lg border border-white/20 flex items-center justify-center"
        >
          <IconComponent size={20} aria-hidden="true" />
        </button>
      </div>
      
      {/* Navigation */}
      <nav className="mt-6">
        <div className="flex flex-wrap justify-center gap-2">
          {navItems.map((item) => (
            <Link
              key={item.path}
              to={item.path}
              className={`px-4 py-2 rounded-lg transition-colors flex items-center gap-2 text-sm font-medium ${
                location.pathname === item.path
                  ? 'bg-white/20 text-white'
                  : 'text-white/80 hover:text-white hover:bg-white/10'
              }`}
            >
              <item.icon size={16} aria-hidden="true" />
              <span>{item.label}</span>
            </Link>
          ))}
        </div>
      </nav>
    </header>
  )
}

export default NavHeader
