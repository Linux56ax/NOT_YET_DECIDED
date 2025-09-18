"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription } from "@/components/ui/alert"

const RouteOptimizer = ({ selectedTrain }) => {
  const [optimizing, setOptimizing] = useState(false)
  const [recommendations, setRecommendations] = useState([])

  const routes = [
    { id: "A-B", traffic: "High", delay: "5 min", alternative: "A-C-B" },
    { id: "B-C", traffic: "Medium", delay: "2 min", alternative: "B-D-C" },
    { id: "C-D", traffic: "Low", delay: "0 min", alternative: null },
    { id: "D-E", traffic: "Medium", delay: "3 min", alternative: "D-C-E" },
    { id: "E-A", traffic: "High", delay: "7 min", alternative: "E-D-A" },
  ]

  const optimizeRoute = async () => {
    setOptimizing(true)

    // Simulate optimization process
    setTimeout(() => {
      const newRecommendations = [
        {
          id: 1,
          trainId: selectedTrain?.id || "T-101",
          currentRoute: "A-B",
          suggestedRoute: "A-C-B",
          timeSaved: "3 min",
          reason: "Lower traffic density",
        },
        {
          id: 2,
          trainId: "T-205",
          currentRoute: "C-D",
          suggestedRoute: "C-B-D",
          timeSaved: "2 min",
          reason: "Signal optimization",
        },
      ]
      setRecommendations(newRecommendations)
      setOptimizing(false)
    }, 2000)
  }

  const applyRoute = (recommendation) => {
    setRecommendations((prev) => prev.filter((rec) => rec.id !== recommendation.id))
    // In a real app, this would send the route change to the train system
  }

  const getTrafficColor = (traffic) => {
    switch (traffic) {
      case "High":
        return "bg-red-500"
      case "Medium":
        return "bg-yellow-500"
      case "Low":
        return "bg-green-500"
      default:
        return "bg-gray-500"
    }
  }

  return (
    <div className="space-y-4">
      {/* Route Status */}
      <div className="space-y-2">
        <h3 className="font-semibold text-foreground">Current Route Status</h3>
        <div className="space-y-2">
          {routes.map((route) => (
            <div key={route.id} className="flex items-center justify-between p-2 rounded border border-border">
              <div className="flex items-center gap-3">
                <span className="font-medium">{route.id}</span>
                <Badge className={`${getTrafficColor(route.traffic)} text-white`}>{route.traffic}</Badge>
              </div>
              <div className="text-sm text-muted-foreground">Delay: {route.delay}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Optimization Controls */}
      <div className="space-y-3">
        <Button onClick={optimizeRoute} disabled={optimizing} className="w-full bg-primary hover:bg-primary/90">
          {optimizing ? "Analyzing Routes..." : "Optimize Routes"}
        </Button>

        {selectedTrain && (
          <Alert>
            <AlertDescription>
              Selected train: <strong>{selectedTrain.id}</strong> on route <strong>{selectedTrain.route}</strong>
            </AlertDescription>
          </Alert>
        )}
      </div>

      {/* Recommendations */}
      {recommendations.length > 0 && (
        <div className="space-y-3">
          <h3 className="font-semibold text-foreground">Route Recommendations</h3>
          {recommendations.map((rec) => (
            <div key={rec.id} className="p-3 rounded-lg border border-primary/20 bg-primary/5">
              <div className="flex items-center justify-between mb-2">
                <span className="font-medium text-foreground">{rec.trainId}</span>
                <Badge variant="outline" className="text-green-600 border-green-600">
                  Save {rec.timeSaved}
                </Badge>
              </div>
              <div className="text-sm text-muted-foreground space-y-1">
                <div>
                  Current: <span className="text-foreground">{rec.currentRoute}</span>
                </div>
                <div>
                  Suggested: <span className="text-primary">{rec.suggestedRoute}</span>
                </div>
                <div>
                  Reason: <span className="text-foreground">{rec.reason}</span>
                </div>
              </div>
              <div className="flex gap-2 mt-3">
                <Button size="sm" onClick={() => applyRoute(rec)} className="bg-primary hover:bg-primary/90">
                  Apply Route
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setRecommendations((prev) => prev.filter((r) => r.id !== rec.id))}
                >
                  Dismiss
                </Button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default RouteOptimizer
