import mongoose from "mongoose";

const trainSchema = new mongoose.Schema({
  name: { type: String, required: true },
  train_no: { type: String, required: true },
  type: { type: String, enum: ["express", "superfast", "local"], default: "express" },
  priority: { type: String, enum: ["low", "medium", "high"], default: "medium" },
  max_speed: Number,
  route: [String],
  rolling_stock_id: String
});

export default mongoose.model("Train", trainSchema);
