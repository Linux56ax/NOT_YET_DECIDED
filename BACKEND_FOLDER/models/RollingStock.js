import mongoose from "mongoose";

const rollingStockSchema = new mongoose.Schema({
  train_id: { type: mongoose.Schema.Types.ObjectId, ref: "Train" },
  locomotive_type: String,
  coaches: Number,
  status: { type: String, default: "operational" },
  last_maintenance: Date
});

export default mongoose.model("RollingStock", rollingStockSchema);
