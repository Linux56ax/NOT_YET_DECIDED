"use client"

import { useState, useEffect } from "react"
import { Badge } from "@/components/ui/badge"

const TrafficMonitor = () => {
  const [trafficData, setTrafficData] = useState([
    { route: "A-B", density: 85, trend: "up", trains: 3 },
    { route: "B-C", density: 45, trend: "stable", trains: 2 },
    { route: "C-D", density: 25, trend: "down", trains: 1 },
    { route: "D-E", density: 60, trend: "up", trains: 2 },
    { route: "E-A", density: 70, trend: "stable", trains: 3 },
  ])

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setTrafficData((prev) =>
        prev.map((route) => ({
          ...route,
          density: Math.max(0, Math.min(100, route.density + (Math.random() - 0.5) * 10)),
          trend: Math.random() > 0.7 ? (Math.random() > 0.5 ? "up" : "down") : route.trend,
        })),
      )
    }, 3000)

    return () => clearInterval(interval)
  }, [])

  const getDensityColor = (density) => {
    if (density >= 80) return "bg-red-500"
    if (density >= 60) return "bg-yellow-500"
    if (density >= 40) return "bg-blue-500"
    return "bg-green-500"
  }

  const getDensityLabel = (density) => {
    if (density >= 80) return "High"
    if (density >= 60) return "Medium"
    if (density >= 40) return "Low"
    return "Clear"
  }

  const getTrendIcon = (trend) => {
    switch (trend) {
      case "up":
        return "↗"
      case "down":
        return "↘"
      case "stable":
        return "→"
      default:
        return "→"
    }
  }

  const getTrendColor = (trend) => {
    switch (trend) {
      case "up":
        return "text-red-500"
      case "down":
        return "text-green-500"
      case "stable":
        return "text-blue-500"
      default:
        return "text-gray-500"
    }
  }

  return (
    <div className="space-y-3">
      <div className="text-sm text-muted-foreground mb-3">Real-time traffic density monitoring</div>

      {trafficData.map((route) => (
        <div key={route.route} className="p-3 rounded-lg border border-border">
          <div className="flex items-center justify-between mb-2">
            <div className="font-medium text-foreground">Route {route.route}</div>
            <div className="flex items-center gap-2">
              <span className={`text-lg ${getTrendColor(route.trend)}`}>{getTrendIcon(route.trend)}</span>
              <Badge className={`${getDensityColor(route.density)} text-white`}>{getDensityLabel(route.density)}</Badge>
            </div>
          </div>

          <div className="space-y-2">
            {/* Density Bar */}
            <div className="w-full bg-muted rounded-full h-2">
              <div
                className={`h-2 rounded-full transition-all duration-500 ${getDensityColor(route.density)}`}
                style={{ width: `${route.density}%` }}
              ></div>
            </div>

            <div className="flex justify-between text-xs text-muted-foreground">
              <span>{route.trains} trains active</span>
              <span>{Math.round(route.density)}% capacity</span>
            </div>
          </div>
        </div>
      ))}

      {/* Summary */}
      <div className="p-3 rounded-lg bg-muted/10 border border-border">
        <div className="text-sm font-medium mb-2">Network Summary</div>
        <div className="grid grid-cols-2 gap-4 text-xs">
          <div>
            <div className="text-muted-foreground">Avg Density</div>
            <div className="font-medium text-foreground">
              {Math.round(trafficData.reduce((acc, route) => acc + route.density, 0) / trafficData.length)}%
            </div>
          </div>
          <div>
            <div className="text-muted-foreground">Total Trains</div>
            <div className="font-medium text-foreground">
              {trafficData.reduce((acc, route) => acc + route.trains, 0)}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default TrafficMonitor
