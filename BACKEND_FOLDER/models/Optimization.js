import mongoose from "mongoose";

const optimizationSchema = new mongoose.Schema({
  scenario: String,
  affected_trains: [String],
  optimized_plan: Object,
  timestamp: { type: Date, default: Date.now }
});

export default mongoose.model("Optimization", optimizationSchema);
