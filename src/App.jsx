 "use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription } from "@/components/ui/alert"
import TrackMap from "@/components/track-map"
import TrainList from "@/components/train-list"
import RouteOptimizer from "@/components/route-optimizer"
import SignalControls from "@/components/signal-controls"
import TrafficMonitor from "@/components/traffic-monitor"

export default function RailwayDashboard() {
  const [selectedTrain, setSelectedTrain] = useState(null)
  const [alerts, setAlerts] = useState([
    { id: 1, type: "warning", message: "High traffic detected on Route A-B", time: "14:32" },
    { id: 2, type: "info", message: "Train T-205 approaching Station Central", time: "14:30" },
  ])

  const [systemStatus, setSystemStatus] = useState({
    totalTrains: 12,
    activeRoutes: 8,
    signalsGreen: 24,
    signalsRed: 3,
    avgDelay: "2.3 min",
  })

  return (
    <div className="min-h-screen bg-background p-4">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-foreground">Railway Control Center</h1>
            <p className="text-muted-foreground">Real-time train tracking and route management</p>
          </div>
          <div className="flex items-center gap-4">
            <Badge variant="outline" className="bg-primary text-primary-foreground">
              System Online
            </Badge>
            <div className="text-sm text-muted-foreground">Last Update: {new Date().toLocaleTimeString()}</div>
          </div>
        </div>

        {/* System Status Cards */}
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Active Trains</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-primary">{systemStatus.totalTrains}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Active Routes</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-primary">{systemStatus.activeRoutes}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Green Signals</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-500">{systemStatus.signalsGreen}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Red Signals</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-red-500">{systemStatus.signalsRed}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium">Avg Delay</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-yellow-500">{systemStatus.avgDelay}</div>
            </CardContent>
          </Card>
        </div>

        {/* Alerts */}
        {alerts.length > 0 && (
          <div className="space-y-2">
            {alerts.map((alert) => (
              <Alert key={alert.id} className={alert.type === "warning" ? "border-yellow-500" : "border-blue-500"}>
                <AlertDescription className="flex items-center justify-between">
                  <span>{alert.message}</span>
                  <span className="text-xs text-muted-foreground">{alert.time}</span>
                </AlertDescription>
              </Alert>
            ))}
          </div>
        )}

        {/* Main Dashboard Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Track Map - Takes up 2 columns */}
          <div className="lg:col-span-2">
            <Card className="h-[600px]">
              <CardHeader>
                <CardTitle>Track Network</CardTitle>
              </CardHeader>
              <CardContent className="h-full">
                <TrackMap selectedTrain={selectedTrain} onTrainSelect={setSelectedTrain} />
              </CardContent>
            </Card>
          </div>

          {/* Right Sidebar */}
          <div className="space-y-6">
            {/* Train List */}
            <Card>
              <CardHeader>
                <CardTitle>Active Trains</CardTitle>
              </CardHeader>
              <CardContent>
                <TrainList selectedTrain={selectedTrain} onTrainSelect={setSelectedTrain} />
              </CardContent>
            </Card>

            {/* Traffic Monitor */}
            <Card>
              <CardHeader>
                <CardTitle>Traffic Monitor</CardTitle>
              </CardHeader>
              <CardContent>
                <TrafficMonitor />
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Bottom Section */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Route Optimizer */}
          <Card>
            <CardHeader>
              <CardTitle>Route Optimizer</CardTitle>
            </CardHeader>
            <CardContent>
              <RouteOptimizer selectedTrain={selectedTrain} />
            </CardContent>
          </Card>

          {/* Signal Controls */}
          <Card>
            <CardHeader>
              <CardTitle>Signal Controls</CardTitle>
            </CardHeader>
            <CardContent>
              <SignalControls />
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
