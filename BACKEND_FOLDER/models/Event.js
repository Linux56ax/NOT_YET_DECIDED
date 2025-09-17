import mongoose from "mongoose";

const eventSchema = new mongoose.Schema({
  station_id: { type: mongoose.Schema.Types.ObjectId, ref: "Station" },
  signal_status: { type: String, enum: ["green", "yellow", "red"], default: "green" },
  timestamp: { type: Date, default: Date.now },
  cause: String
});

export default mongoose.model("Event", eventSchema);
