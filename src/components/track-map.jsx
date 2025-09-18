"use client"

import { useState, useEffect } from "react"

const TrackMap = ({ selectedTrain, onTrainSelect }) => {
  const [trains, setTrains] = useState([
    { id: "T-101", x: 150, y: 200, status: "moving", route: "A-B", speed: 65 },
    { id: "T-205", x: 400, y: 150, status: "stopped", route: "C-D", speed: 0 },
    { id: "T-308", x: 300, y: 350, status: "moving", route: "B-C", speed: 45 },
    { id: "T-412", x: 550, y: 250, status: "moving", route: "D-E", speed: 70 },
  ])

  const [signals, setSignals] = useState([
    { id: "S1", x: 200, y: 180, status: "green" },
    { id: "S2", x: 350, y: 130, status: "red" },
    { id: "S3", x: 450, y: 300, status: "green" },
    { id: "S4", x: 500, y: 200, status: "yellow" },
  ])

  // Simulate train movement
  useEffect(() => {
    const interval = setInterval(() => {
      setTrains((prevTrains) =>
        prevTrains.map((train) => {
          if (train.status === "moving") {
            return {
              ...train,
              x: train.x + (Math.random() - 0.5) * 2,
              y: train.y + (Math.random() - 0.5) * 2,
            }
          }
          return train
        }),
      )
    }, 1000)

    return () => clearInterval(interval)
  }, [])

  const getTrainColor = (train) => {
    if (selectedTrain?.id === train.id) return "#84cc16"
    return train.status === "moving" ? "#10b981" : "#ef4444"
  }

  const getSignalColor = (status) => {
    switch (status) {
      case "green":
        return "#10b981"
      case "red":
        return "#ef4444"
      case "yellow":
        return "#f59e0b"
      default:
        return "#6b7280"
    }
  }

  return (
    <div className="relative w-full h-full bg-card rounded-lg overflow-hidden">
      <svg width="100%" height="100%" viewBox="0 0 600 400" className="absolute inset-0">
        {/* Track Lines */}
        <g stroke="#ffffff" strokeWidth="2" fill="none">
          {/* Main horizontal tracks */}
          <line x1="50" y1="200" x2="550" y2="200" />
          <line x1="50" y1="150" x2="550" y2="150" />
          <line x1="50" y1="250" x2="550" y2="250" />

          {/* Connecting tracks */}
          <line x1="200" y1="150" x2="200" y2="250" />
          <line x1="350" y1="150" x2="350" y2="250" />
          <line x1="450" y1="150" x2="450" y2="250" />

          {/* Junction curves */}
          <path d="M 200 200 Q 220 180 240 200" />
          <path d="M 350 200 Q 370 180 390 200" />
          <path d="M 200 200 Q 220 220 240 200" />
          <path d="M 350 200 Q 370 220 390 200" />
        </g>

        {/* Stations */}
        <g>
          <rect x="45" y="145" width="10" height="60" fill="#6b7280" />
          <rect x="195" y="145" width="10" height="60" fill="#6b7280" />
          <rect x="345" y="145" width="10" height="60" fill="#6b7280" />
          <rect x="445" y="145" width="10" height="60" fill="#6b7280" />
          <rect x="545" y="145" width="10" height="60" fill="#6b7280" />

          {/* Station labels */}
          <text x="50" y="130" fill="#ffffff" fontSize="12" textAnchor="middle">
            A
          </text>
          <text x="200" y="130" fill="#ffffff" fontSize="12" textAnchor="middle">
            B
          </text>
          <text x="350" y="130" fill="#ffffff" fontSize="12" textAnchor="middle">
            C
          </text>
          <text x="450" y="130" fill="#ffffff" fontSize="12" textAnchor="middle">
            D
          </text>
          <text x="550" y="130" fill="#ffffff" fontSize="12" textAnchor="middle">
            E
          </text>
        </g>

        {/* Signals */}
        {signals.map((signal) => (
          <g key={signal.id}>
            <circle
              cx={signal.x}
              cy={signal.y}
              r="6"
              fill={getSignalColor(signal.status)}
              stroke="#ffffff"
              strokeWidth="1"
            />
            <text x={signal.x} y={signal.y - 15} fill="#ffffff" fontSize="10" textAnchor="middle">
              {signal.id}
            </text>
          </g>
        ))}

        {/* Trains */}
        {trains.map((train) => (
          <g key={train.id} onClick={() => onTrainSelect(train)} className="cursor-pointer">
            <rect
              x={train.x - 15}
              y={train.y - 8}
              width="30"
              height="16"
              fill={getTrainColor(train)}
              stroke="#ffffff"
              strokeWidth="1"
              rx="2"
            />
            <text x={train.x} y={train.y + 3} fill="#ffffff" fontSize="10" textAnchor="middle" fontWeight="bold">
              {train.id}
            </text>
            {/* Speed indicator */}
            {train.status === "moving" && (
              <text x={train.x} y={train.y - 15} fill="#84cc16" fontSize="8" textAnchor="middle">
                {train.speed}km/h
              </text>
            )}
          </g>
        ))}
      </svg>

      {/* Legend */}
      <div className="absolute bottom-4 left-4 bg-card/90 p-3 rounded-lg border border-border">
        <div className="text-xs space-y-1">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-green-500 rounded"></div>
            <span>Moving Train</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-red-500 rounded"></div>
            <span>Stopped Train</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-primary rounded"></div>
            <span>Selected Train</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default TrackMap
