import { PageHeader, Card, StatsCard, Button } from '../shared/ui'

const Quality = () => {
  const qualityMetrics = [
    {
      title: 'Aspect Ratio',
      value: '1.42',
      description: 'Average element aspect ratio',
      icon: '📐',
      trend: '-0.05',
      trendDirection: 'up'
    },
    {
      title: 'Skewness',
      value: '0.18',
      description: 'Average element skewness',
      icon: '📊',
      trend: '-0.02',
      trendDirection: 'up'
    },
    {
      title: 'Min Angle',
      value: '28.5°',
      description: 'Minimum element angle',
      icon: '📏',
      trend: '+1.2°',
      trendDirection: 'up'
    },
    {
      title: 'Quality Score',
      value: '89.2%',
      description: 'Overall mesh quality',
      icon: '⭐',
      trend: '+2.1%',
      trendDirection: 'up'
    }
  ]

  return (
    <div className="max-w-7xl mx-auto p-6">
      <PageHeader
        title="Quality Analysis"
        subtitle="Analyze and evaluate mesh quality metrics and performance indicators for generated meshes."
        icon="⭐"
        backLink={{ href: '/', label: 'Back to Dashboard' }}
        actions={[
          <Button variant="primary" size="sm">
            Run Analysis
          </Button>,
          <Button variant="outline" size="sm">
            Export Report
          </Button>
        ]}
      />

      {/* Quality Metrics */}
      <div className="mb-8">
        <h3 className="text-xl font-semibold text-text-primary mb-6">Current Metrics</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {qualityMetrics.map((metric, index) => (
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
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <Card title="Quality Distribution" className="p-6">
          <div className="h-64 bg-bg-secondary rounded-lg flex items-center justify-center">
            <div className="text-center text-text-secondary">
              <div className="text-4xl mb-2">📊</div>
              <div className="text-sm">Quality distribution chart</div>
              <div className="text-xs mt-1">Coming soon</div>
            </div>
          </div>
        </Card>

        <Card title="Mesh Statistics" className="p-6">
          <div className="space-y-4">
            <div className="flex justify-between items-center py-2 border-b border-border-custom">
              <span className="text-text-secondary">Total Elements:</span>
              <span className="text-text-primary font-medium">2,847</span>
            </div>
            <div className="flex justify-between items-center py-2 border-b border-border-custom">
              <span className="text-text-secondary">Total Vertices:</span>
              <span className="text-text-primary font-medium">1,523</span>
            </div>
            <div className="flex justify-between items-center py-2 border-b border-border-custom">
              <span className="text-text-secondary">Boundary Edges:</span>
              <span className="text-text-primary font-medium">142</span>
            </div>
            <div className="flex justify-between items-center py-2 border-b border-border-custom">
              <span className="text-text-secondary">Area Coverage:</span>
              <span className="text-text-primary font-medium">98.7%</span>
            </div>
            <div className="flex justify-between items-center py-2">
              <span className="text-text-secondary">Mesh Density:</span>
              <span className="text-text-primary font-medium">High</span>
            </div>
          </div>
        </Card>
      </div>

      {/* Analysis Reports */}
      <Card title="Analysis Reports" className="p-6">
        <div className="space-y-4">
          <div className="flex items-center justify-between p-4 bg-bg-secondary rounded-lg">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-green-500/10 text-green-500 rounded-full flex items-center justify-center">
                ✓
              </div>
              <div>
                <div className="font-medium text-text-primary">Quality Report - March 2024</div>
                <div className="text-sm text-text-secondary">Generated 2 hours ago</div>
              </div>
            </div>
            <Button variant="outline" size="sm">
              Download
            </Button>
          </div>

          <div className="flex items-center justify-between p-4 bg-bg-secondary rounded-lg">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-blue-500/10 text-blue-500 rounded-full flex items-center justify-center">
                📊
              </div>
              <div>
                <div className="font-medium text-text-primary">Mesh Comparison Analysis</div>
                <div className="text-sm text-text-secondary">Generated yesterday</div>
              </div>
            </div>
            <Button variant="outline" size="sm">
              View
            </Button>
          </div>

          <div className="flex items-center justify-between p-4 bg-bg-secondary rounded-lg">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-yellow-500/10 text-yellow-500 rounded-full flex items-center justify-center">
                📋
              </div>
              <div>
                <div className="font-medium text-text-primary">Performance Benchmark</div>
                <div className="text-sm text-text-secondary">Generated 3 days ago</div>
              </div>
            </div>
            <Button variant="outline" size="sm">
              View
            </Button>
          </div>
        </div>
      </Card>
    </div>
  )
}

export default Quality
