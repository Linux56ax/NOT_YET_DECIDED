import mongoose from "mongoose";

const stationSchema = new mongoose.Schema({
  name: { type: String, required: true },
  code: { type: String, required: true },
  type: { type: String, enum: ["station", "junction"], default: "station" },
  lat: Number,
  lon: Number,
  platforms: Number,
  connections: [String]
});

export default mongoose.model("Station", stationSchema);
