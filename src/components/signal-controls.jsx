"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"

const SignalControls = () => {
  const [signals, setSignals] = useState([
    { id: "S1", location: "Junction A-B", status: "green", auto: true },
    { id: "S2", location: "Junction B-C", status: "red", auto: false },
    { id: "S3", location: "Junction C-D", status: "green", auto: true },
    { id: "S4", location: "Junction D-E", status: "yellow", auto: true },
    { id: "S5", location: "Platform A", status: "green", auto: true },
    { id: "S6", location: "Platform B", status: "red", auto: false },
  ])

  const toggleSignal = (signalId) => {
    setSignals((prev) =>
      prev.map((signal) => {
        if (signal.id === signalId) {
          let newStatus
          if (signal.status === "green") newStatus = "yellow"
          else if (signal.status === "yellow") newStatus = "red"
          else newStatus = "green"

          return { ...signal, status: newStatus, auto: false }
        }
        return signal
      }),
    )
  }

  const toggleAutoMode = (signalId) => {
    setSignals((prev) =>
      prev.map((signal) => {
        if (signal.id === signalId) {
          return { ...signal, auto: !signal.auto }
        }
        return signal
      }),
    )
  }

  const getStatusColor = (status) => {
    switch (status) {
      case "green":
        return "bg-green-500"
      case "yellow":
        return "bg-yellow-500"
      case "red":
        return "bg-red-500"
      default:
        return "bg-gray-500"
    }
  }

  const emergencyStop = () => {
    setSignals((prev) =>
      prev.map((signal) => ({
        ...signal,
        status: "red",
        auto: false,
      })),
    )
  }

  const resetToAuto = () => {
    setSignals((prev) =>
      prev.map((signal) => ({
        ...signal,
        auto: true,
      })),
    )
  }

  return (
    <div className="space-y-4">
      {/* Emergency Controls */}
      <div className="flex gap-2">
        <Button onClick={emergencyStop} variant="destructive" className="flex-1">
          Emergency Stop All
        </Button>
        <Button onClick={resetToAuto} variant="outline" className="flex-1 bg-transparent">
          Reset to Auto
        </Button>
      </div>

      {/* Signal List */}
      <div className="space-y-3 max-h-80 overflow-y-auto">
        {signals.map((signal) => (
          <div key={signal.id} className="p-3 rounded-lg border border-border">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-3">
                <div className={`w-4 h-4 rounded-full ${getStatusColor(signal.status)}`}></div>
                <div>
                  <div className="font-medium text-foreground">{signal.id}</div>
                  <div className="text-xs text-muted-foreground">{signal.location}</div>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <Badge variant={signal.auto ? "default" : "secondary"}>{signal.auto ? "AUTO" : "MANUAL"}</Badge>
                <Badge className={`${getStatusColor(signal.status)} text-white uppercase`}>{signal.status}</Badge>
              </div>
            </div>

            <div className="flex gap-2">
              <Button
                size="sm"
                variant="outline"
                onClick={() => toggleSignal(signal.id)}
                disabled={signal.auto}
                className="flex-1"
              >
                Change Signal
              </Button>
              <Button
                size="sm"
                variant={signal.auto ? "secondary" : "default"}
                onClick={() => toggleAutoMode(signal.id)}
                className="flex-1"
              >
                {signal.auto ? "Manual" : "Auto"}
              </Button>
            </div>
          </div>
        ))}
      </div>

      {/* Signal Legend */}
      <div className="p-3 rounded-lg bg-muted/10 border border-border">
        <div className="text-sm font-medium mb-2">Signal Status</div>
        <div className="grid grid-cols-3 gap-2 text-xs">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-green-500"></div>
            <span>Go</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
            <span>Caution</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-red-500"></div>
            <span>Stop</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default SignalControls
