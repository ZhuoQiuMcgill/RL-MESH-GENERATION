import { PageHeader, Card, StatsCard, Button } from '../shared/ui'

const Angle = () => {
  const angleMetrics = [
    {
      title: 'Min Angle',
      value: '28.5°',
      description: 'Smallest element angle',
      icon: '📊',
      trend: '+1.2°',
      trendDirection: 'up'
    },
    {
      title: 'Max Angle',
      value: '142.3°',
      description: 'Largest element angle',
      icon: '📊',
      trend: '-2.1°',
      trendDirection: 'up'
    },
    {
      title: 'Avg Angle',
      value: '60.8°',
      description: 'Average element angle',
      icon: '📊',
      trend: '+0.3°',
      trendDirection: 'up'
    },
    {
      title: 'Angle Quality',
      value: '87.4%',
      description: 'Angle distribution quality',
      icon: '⭐',
      trend: '+1.8%',
      trendDirection: 'up'
    }
  ]

  const analysisTools = [
    {
      title: 'Angle Histogram',
      description: 'Visualize the distribution of element angles across the mesh.',
      icon: '📊'
    },
    {
      title: 'Quality Assessment',
      description: 'Evaluate mesh quality based on angle criteria and standards.',
      icon: '⭐'
    },
    {
      title: 'Topology Analysis',
      description: 'Analyze mesh connectivity and topological properties.',
      icon: '🔗'
    },
    {
      title: 'Angle Mapping',
      description: 'Visualize angle distribution with color-coded mapping.',
      icon: '🎨'
    }
  ]

  return (
    <div className="max-w-7xl mx-auto p-6">
      <PageHeader
        title="Angle Analysis"
        subtitle="Analyze mesh angles, topology, and geometric properties to assess mesh quality and identify potential issues."
        icon="📊"
        backLink={{ href: '/', label: 'Back to Dashboard' }}
        actions={[
          <Button variant="primary" size="sm">
            Analyze Mesh
          </Button>,
          <Button variant="outline" size="sm">
            Export Report
          </Button>
        ]}
      />

      {/* Angle Metrics */}
      <div className="mb-8">
        <h3 className="text-xl font-semibold text-text-primary mb-6">Angle Statistics</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {angleMetrics.map((metric, index) => (
            <StatsCard
              key={index}
              title={metric.title}
              value={metric.value}
              description={metric.description}
              icon={metric.icon}
              trend={metric.trend}
              trendDirection={metric.trendDirection}
            />
          ))}
        </div>
      </div>

      {/* Analysis Tools */}
      <div className="mb-8">
        <h3 className="text-xl font-semibold text-text-primary mb-6">Analysis Tools</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {analysisTools.map((tool, index) => (
            <Card key={index} className="p-6">
              <div className="flex items-start gap-4">
                <div className="text-3xl flex-shrink-0">
                  {tool.icon}
                </div>
                <div className="flex-1">
                  <h4 className="text-lg font-semibold text-text-primary mb-2">{tool.title}</h4>
                  <p className="text-text-secondary text-sm mb-4">
                    {tool.description}
                  </p>
                  <Button variant="outline" size="sm" className="w-full">
                    Launch Tool
                  </Button>
                </div>
              </div>
            </Card>
          ))}
        </div>
      </div>

      {/* Angle Distribution Visualization */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card title="Angle Distribution" className="p-6">
          <div className="h-64 bg-bg-secondary rounded-lg flex items-center justify-center">
            <div className="text-center text-text-secondary">
              <div className="text-4xl mb-2">📊</div>
              <div className="text-sm">Angle distribution chart</div>
              <div className="text-xs mt-1">Coming soon</div>
            </div>
          </div>
        </Card>

        <Card title="Quality Breakdown" className="p-6">
          <div className="space-y-4">
            <div className="flex justify-between items-center py-2 border-b border-border-custom">
              <span className="text-text-secondary">Excellent (&gt; 30°):</span>
              <div className="flex items-center gap-2">
                <div className="w-20 h-2 bg-bg-secondary rounded-full overflow-hidden">
                  <div className="w-4/5 h-full bg-green-500"></div>
                </div>
                <span className="text-text-primary font-medium text-sm">82%</span>
              </div>
            </div>
            <div className="flex justify-between items-center py-2 border-b border-border-custom">
              <span className="text-text-secondary">Good (20-30°):</span>
              <div className="flex items-center gap-2">
                <div className="w-20 h-2 bg-bg-secondary rounded-full overflow-hidden">
                  <div className="w-1/4 h-full bg-yellow-500"></div>
                </div>
                <span className="text-text-primary font-medium text-sm">15%</span>
              </div>
            </div>
            <div className="flex justify-between items-center py-2">
              <span className="text-text-secondary">Poor (&lt; 20°):</span>
              <div className="flex items-center gap-2">
                <div className="w-20 h-2 bg-bg-secondary rounded-full overflow-hidden">
                  <div className="w-1/12 h-full bg-red-500"></div>
                </div>
                <span className="text-text-primary font-medium text-sm">3%</span>
              </div>
            </div>
          </div>
        </Card>
      </div>
    </div>
  )
}

export default Angle
