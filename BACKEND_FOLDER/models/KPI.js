import mongoose from "mongoose";

const kpiSchema = new mongoose.Schema({
  date: { type: Date, default: Date.now },
  total_trains: Number,
  avg_delay: Number,
  on_time_percent: Number,
  throughput: Number,
  utilization: Number
});

export default mongoose.model("KPI", kpiSchema);
