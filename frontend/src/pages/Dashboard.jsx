import { Link } from 'react-router-dom'
import { Card, StatsCard, PageHeader } from '../shared/ui'
import { memo, useMemo } from 'react'

const Dashboard = memo(() => {
  // Memoize static data to prevent recreation on re-renders
  const moduleCards = useMemo(() => [
    {
      title: 'Training',
      description: 'Start or monitor training sessions',
      link: '/train',
      icon: '🚂',
    },
    {
      title: 'History',
      description: 'View training history and logs',
      link: '/history',
      icon: '📋',
    },
    {
      title: 'Quality Analysis',
      description: 'Analyze mesh quality metrics',
      link: '/quality',
      icon: '⭐',
    },
    {
      title: 'Geometry Tools',
      description: 'Geometry manipulation tools',
      link: '/geometry',
      icon: '📐',
    },
    {
      title: 'Canvas',
      description: 'Interactive 3D mesh canvas',
      link: '/canvas',
      icon: '🎨',
    },
    {
      title: 'Angle Analysis',
      description: 'Analyze mesh angles and topology',
      link: '/angle',
      icon: '📊',
    },
    {
      title: 'Action Spaces',
      description: 'Configure RL action spaces',
      link: '/action',
      icon: '⚡',
    },
    {
      title: 'Generator',
      description: 'Mesh generation tools',
      link: '/generator',
      icon: '🔧',
    }
  ], [])

  const statsData = useMemo(() => [
    {
      title: 'Training Episodes',
      value: '156',
      description: 'Total completed episodes',
      icon: '📊',
      trend: '+12%',
      trendDirection: 'up'
    },
    {
      title: 'Quality Score',
      value: '89.2%',
      description: 'Average mesh quality',
      icon: '⭐',
      trend: '+2.1%',
      trendDirection: 'up'
    },
    {
      title: 'Generated Meshes',
      value: '1,247',
      description: 'Total meshes created',
      icon: '🔧',
      trend: '+23',
      trendDirection: 'up'
    },
    {
      title: 'Active Models',
      value: '4',
      description: 'Models ready for training',
      icon: '🧠',
      trendDirection: 'neutral'
    }
  ], [])

  return (
    <div className="max-w-7xl mx-auto p-6">
      <PageHeader
        title="RL Mesh Generation"
        subtitle="Explore our reinforcement learning tools for 3D mesh generation and analysis."
        icon="🤖"
      />

      {/* Statistics Cards */}
      <div className="mb-12">
        <h3 className="text-xl font-semibold text-text-primary mb-6">Overview</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {statsData.map((stat, index) => (
            <StatsCard
              key={index}
              title={stat.title}
              value={stat.value}
              description={stat.description}
              icon={stat.icon}
              trend={stat.trend}
              trendDirection={stat.trendDirection}
            />
          ))}
        </div>
      </div>

      {/* Module Cards */}
      <div className="mb-8">
        <h3 className="text-xl font-semibold text-text-primary mb-6">Modules</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {moduleCards.map((card, index) => (
            <Link key={index} to={card.link} className="block">
              <Card className="h-full group hover:shadow-lg transition-all duration-200 hover:scale-[1.02] cursor-pointer">
                <div className="flex items-center gap-4 mb-4">
                  <div className="text-3xl flex-shrink-0">
                    {card.icon}
                  </div>
                  <div className="min-w-0">
                    <h4 className="text-lg font-semibold text-text-primary group-hover:text-accent transition-colors truncate">
                      {card.title}
                    </h4>
                  </div>
                </div>
                <p className="text-text-secondary text-sm leading-relaxed mb-4">
                  {card.description}
                </p>
                <div className="flex items-center text-accent text-sm font-medium mt-auto">
                  Explore
                  <svg className="w-4 h-4 ml-2 transition-transform group-hover:translate-x-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </div>
              </Card>
            </Link>
          ))}
        </div>
      </div>

      {/* Recent Activity */}
      <Card title="Recent Activity" className="mb-8">
        <div className="space-y-4">
          <div className="flex items-center justify-between py-3 border-b border-border-custom last:border-b-0">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-blue-500/10 text-blue-500 rounded-full flex items-center justify-center text-sm font-medium">
                T
              </div>
              <div>
                <div className="text-sm font-medium text-text-primary">Training session completed</div>
                <div className="text-xs text-text-secondary">PPO model on simple_square mesh</div>
              </div>
            </div>
            <div className="text-xs text-text-secondary">2 hours ago</div>
          </div>
          <div className="flex items-center justify-between py-3 border-b border-border-custom last:border-b-0">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-green-500/10 text-green-500 rounded-full flex items-center justify-center text-sm font-medium">
                M
              </div>
              <div>
                <div className="text-sm font-medium text-text-primary">New mesh generated</div>
                <div className="text-xs text-text-secondary">Quality score: 92.4%</div>
              </div>
            </div>
            <div className="text-xs text-text-secondary">4 hours ago</div>
          </div>
          <div className="flex items-center justify-between py-3">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-yellow-500/10 text-yellow-500 rounded-full flex items-center justify-center text-sm font-medium">
                A
              </div>
              <div>
                <div className="text-sm font-medium text-text-primary">Analysis completed</div>
                <div className="text-xs text-text-secondary">Angle distribution report generated</div>
              </div>
            </div>
            <div className="text-xs text-text-secondary">6 hours ago</div>
          </div>
        </div>
      </Card>
    </div>
  )
})

Dashboard.displayName = 'Dashboard'

export default Dashboard
