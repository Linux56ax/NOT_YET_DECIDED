"use client"

import { Badge } from "@/components/ui/badge"

const TrainList = ({ selectedTrain, onTrainSelect }) => {
  const trains = [
    { id: "T-101", route: "A-B", status: "On Time", delay: 0, speed: 65, nextStation: "Station B", eta: "14:45" },
    { id: "T-205", route: "C-D", status: "Delayed", delay: 5, speed: 0, nextStation: "Station D", eta: "15:10" },
    { id: "T-308", route: "B-C", status: "On Time", delay: 0, speed: 45, nextStation: "Station C", eta: "14:52" },
    { id: "T-412", route: "D-E", status: "Early", delay: -2, speed: 70, nextStation: "Station E", eta: "14:38" },
    { id: "T-515", route: "E-A", status: "On Time", delay: 0, speed: 55, nextStation: "Station A", eta: "15:15" },
    { id: "T-620", route: "A-C", status: "Delayed", delay: 8, speed: 35, nextStation: "Station C", eta: "15:25" },
  ]

  const getStatusColor = (status) => {
    switch (status) {
      case "On Time":
        return "bg-green-500"
      case "Delayed":
        return "bg-red-500"
      case "Early":
        return "bg-blue-500"
      default:
        return "bg-gray-500"
    }
  }

  return (
    <div className="space-y-3 max-h-96 overflow-y-auto">
      {trains.map((train) => (
        <div
          key={train.id}
          className={`p-3 rounded-lg border cursor-pointer transition-colors ${
            selectedTrain?.id === train.id ? "border-primary bg-primary/10" : "border-border hover:border-primary/50"
          }`}
          onClick={() => onTrainSelect(train)}
        >
          <div className="flex items-center justify-between mb-2">
            <div className="font-semibold text-foreground">{train.id}</div>
            <Badge className={`${getStatusColor(train.status)} text-white`}>{train.status}</Badge>
          </div>

          <div className="text-sm text-muted-foreground space-y-1">
            <div className="flex justify-between">
              <span>Route:</span>
              <span className="text-foreground">{train.route}</span>
            </div>
            <div className="flex justify-between">
              <span>Speed:</span>
              <span className="text-foreground">{train.speed} km/h</span>
            </div>
            <div className="flex justify-between">
              <span>Next:</span>
              <span className="text-foreground">{train.nextStation}</span>
            </div>
            <div className="flex justify-between">
              <span>ETA:</span>
              <span className="text-foreground">{train.eta}</span>
            </div>
            {train.delay !== 0 && (
              <div className="flex justify-between">
                <span>Delay:</span>
                <span className={train.delay > 0 ? "text-red-500" : "text-blue-500"}>
                  {train.delay > 0 ? "+" : ""}
                  {train.delay} min
                </span>
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  )
}

export default TrainList
